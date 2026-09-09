"""
Pre-flight, health and rerun helpers for the cross-model tournament.

Everything here answers one question: **is a matchup real?** A tournament matchup can
exit cleanly, write a complete full-length JSON, and still be void. That is not
hypothetical — before the 260729 fix a missing `environment_parameters` block made the
build throw on every action, which froze CompetitiveAgent2 at its spawn and handed P1
every trial in all 64 matchups, twice, with nothing in the exit status to show it.

So the checks look at behaviour, not at exit codes:
  - both agents must occupy more than one position (an agent that never moves is frozen),
  - both agents must trigger some walls (one player winning everything is the same bug),
  - the trial count must be close to the requested episode count (a short run stalled).

The per-batch config, narrative and results stay in the notebooks
(`tournament/run_tournament*.ipynb`); everything that computes or asserts lives here.
Checks read the raw session JSON by regex rather than going through octagon_analysis,
so this module stays importable in the mlagents env with no analysis dependencies.
"""

from __future__ import annotations

import collections
import re
import shutil
import tempfile
import time
from itertools import combinations_with_replacement
from pathlib import Path

from tournament.tournament_func import run_matchup, write_tournament_config_yaml

# A run is called SHORT when it logs more than this many trials fewer than requested.
# Sampled actions mean a run occasionally finishes a trial or two under; a real stall
# (the 260715_0002 self-matchup logged 68 of ~350) is nowhere near this tolerance.
TRIAL_SHORTFALL_TOLERANCE = 5


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

def preflight(models_root, unity_env_path, trial_seq, run_ids=None,
              include_mirrors=True, include_self_matchups=True, verbose=True):
    """
    Check a batch can run at all, and report the matchup count. Raises on anything wrong.

    Takes the batch config; returns the number of matchups the round-robin will run.

    The important check is the last one: generate a throwaway tournament config and
    confirm `environment_parameters` / `step_penalty` is REALLY in it. This has to run
    from the notebook's own call into a freshly reloaded module, because a stale kernel
    holding pre-fix code would not carry an assertion inside tournament_func.py either.
    """
    models_root, unity_env_path, trial_seq = Path(models_root), Path(unity_env_path), Path(trial_seq)
    if not unity_env_path.exists():
        raise FileNotFoundError(f"Build not found: {unity_env_path} (set UNITY_ENV_PATH to your fresh tournament build)")
    if not models_root.exists():
        raise FileNotFoundError(f"MODELS_ROOT not found: {models_root}")
    if not trial_seq.exists():
        raise FileNotFoundError(f"TRIAL_SEQ not found: {trial_seq}")

    if run_ids is None:
        run_ids = sorted(d.name for d in models_root.iterdir()
                         if d.is_dir() and any(d.glob("*/checkpoint.pt")))
    for rid in run_ids:
        if not list((models_root / rid).glob("*/checkpoint.pt")):
            raise FileNotFoundError(f"No <behaviour>/checkpoint.pt under {models_root / rid}")

    # Stale-kernel guard: without environment_parameters the build throws on every
    # action, CompetitiveAgent2 never gets stepped, and every matchup is a walkover
    # that still writes a complete, healthy-looking JSON.
    with tempfile.TemporaryDirectory() as tmp:
        probe = write_tournament_config_yaml(models_root / run_ids[0], Path(tmp) / "probe.yaml")
        txt = probe.read_text()
    if "environment_parameters" not in txt or "step_penalty" not in txt:
        raise RuntimeError(
            "Generated config has NO environment_parameters/step_penalty — the build will "
            "throw every action and freeze P2. Your kernel is running pre-fix code: restart "
            "it and re-run from the top."
        )

    n_matchups = len(expected_matchups(run_ids, include_mirrors, include_self_matchups))
    if verbose:
        print(f"All paths OK, config carries step_penalty. {len(run_ids)} entrants "
              f"=> {n_matchups} matchups ({'with' if include_mirrors else 'no'} mirrors, "
              f"{'incl.' if include_self_matchups else 'no'} self-matchups).")
    return n_matchups


def expected_matchups(run_ids, include_mirrors=True, include_self_matchups=True):
    """Return the ordered (P1, P2) pairs run_tournament will run, in the same order."""
    matchups = []
    for a, b in combinations_with_replacement(run_ids, 2):
        if a == b:
            if include_self_matchups:
                matchups.append((a, b))
        else:
            matchups.append((a, b))
            if include_mirrors:
                matchups.append((b, a))
    return matchups


# ---------------------------------------------------------------------------
# Health of a finished matchup
# ---------------------------------------------------------------------------

def newest_session_json(matchup_dir):
    """Return the most recently written session JSON in a matchup dir, or None."""
    jsons = [p for p in Path(matchup_dir).glob("*.json")
             if not p.name.startswith("._") and "config" not in p.name]
    return max(jsons, key=lambda p: p.stat().st_mtime) if jsons else None


def matchup_health(matchup_dir, expected_trials=None, octagon_dir=None, tournament_id=None):
    """
    Return a health dict for one finished matchup, without loading the analysis stack.

    Takes the matchup output dir; optionally the requested episode count (to flag SHORT
    runs) and the Octagon dir + tournament id (to count step_penalty exceptions in the
    Unity player log). Keys: n_trials, uniq_positions, triggers, n_exceptions, flags, ok.
    """
    matchup_dir = Path(matchup_dir)
    log = newest_session_json(matchup_dir)
    if log is None:
        return dict(json=None, n_trials=0, uniq_positions={0: 0, 1: 0}, triggers={0: 0, 1: 0},
                    n_exceptions=None, flags=["NO JSON WRITTEN"], ok=False)

    raw = log.read_text()
    n_trials = raw.count('"trial start"')
    # distinct logged positions per agent; a frozen agent sits on a single one
    uniq = {c: len(set(re.findall(rf'"clientId":{c},"location":\{{[^}}]*\}}', raw))) for c in (0, 1)}
    trig_counts = collections.Counter(re.findall(r'"triggerClient":(\d)', raw))
    triggers = {c: trig_counts.get(str(c), 0) for c in (0, 1)}

    n_exceptions = None
    if octagon_dir is not None:
        tid = tournament_id or matchup_dir.name
        player_log = Path(octagon_dir) / "results" / "tournament" / tid / "run_logs" / "Player-0.log"
        if player_log.exists():
            n_exceptions = player_log.read_text(errors="ignore").count("step_penalty not found")

    flags = []
    if expected_trials is not None and n_trials < expected_trials - TRIAL_SHORTFALL_TOLERANCE:
        flags.append(f"SHORT (expected ~{expected_trials - 1}, got {n_trials})")
    if min(uniq.values()) <= 1:
        flags.append("AGENT FROZEN — matchup is void")
    if min(triggers.values()) == 0:
        flags.append("one player won every trial")
    if n_exceptions:
        flags.append(f"{n_exceptions} step_penalty exceptions in the Unity log")

    return dict(json=log, n_trials=n_trials, uniq_positions=uniq, triggers=triggers,
                n_exceptions=n_exceptions, flags=flags, ok=not flags)


def format_health(tournament_id, health):
    """One-line summary of a matchup_health dict, for printing."""
    if health["json"] is None:
        return f"{tournament_id}: NO JSON WRITTEN"
    u, t = health["uniq_positions"], health["triggers"]
    tail = "  OK" if health["ok"] else "  <-- " + "; ".join(health["flags"])
    return (f"{tournament_id}: {health['n_trials']} trials | unique positions "
            f"P1 {u[0]}, P2 {u[1]} | triggers P1 {t[0]}, P2 {t[1]}{tail}")


def check_batch(out_root, tournament_ids, expected_trials=None, octagon_dir=None,
                logs=None, verbose=True):
    """
    Validate a whole batch. Takes the batch output dir and the matchup ids to check
    (usually `logs.keys()`); returns the list of suspect ids.

    Passing `logs` (the dict run_tournament returned) also reports which matchups raised.
    Only suspect matchups are printed — a clean batch prints just the two summary lines.
    """
    out_root = Path(out_root)
    tournament_ids = list(tournament_ids)

    if logs is not None:
        failed = [k for k, v in logs.items() if v is None]
        if verbose:
            print(f"{len(logs) - len(failed)}/{len(logs)} matchups ran without error")
            for k in failed:
                print("  RAISED:", k)

    suspect = []
    if verbose:
        print("\nValidity check (both agents must move, both must win some trials):")
    for tid in tournament_ids:
        health = matchup_health(out_root / tid, expected_trials=expected_trials,
                                octagon_dir=octagon_dir, tournament_id=tid)
        if not health["ok"]:
            suspect.append(tid)
            if verbose:
                print("  " + format_health(tid, health))
    if verbose:
        print(f"\n{len(tournament_ids) - len(suspect)}/{len(tournament_ids)} matchups look valid."
              + ("" if not suspect else f" Investigate before analysing: {suspect}"))
    return suspect


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test(model_a_run_dir, model_b_run_dir, octagon_dir, unity_env_path, out_root,
               trial_seq, episodes=8, config_template_run_dir=None, base_port=5015,
               verbose=True, **matchup_kwargs):
    """
    Run one short matchup and verdict it, before committing to a full round-robin.

    Takes the two entrant run dirs and the batch config; writes into `out_root/_smoke_test`
    and returns the health dict. Raises if an agent is frozen or Unity is throwing on
    step_penalty — the two failures that silently void a whole batch.
    """
    out_root = Path(out_root)
    config_yaml = write_tournament_config_yaml(
        template_run_dir=Path(config_template_run_dir or model_a_run_dir),
        out_yaml=out_root / "tournament_config.yaml",
    )
    log = run_matchup(
        model_a_run_dir=model_a_run_dir,
        model_b_run_dir=model_b_run_dir,
        octagon_dir=octagon_dir,
        unity_env_path=unity_env_path,
        out_path=out_root / "_smoke_test",
        episodes=episodes,
        config_yaml=config_yaml,
        trial_seq=trial_seq,
        base_port=base_port,
        **matchup_kwargs,
    )
    if verbose:
        print("smoke-test log:", log)

    # The smoke test is short, so don't flag it as SHORT — only frozen/one-sided matter.
    tid = f"{Path(model_a_run_dir).name}__vs__{Path(model_b_run_dir).name}"
    health = matchup_health(out_root / "_smoke_test", octagon_dir=octagon_dir, tournament_id=tid)
    if verbose:
        u, t = health["uniq_positions"], health["triggers"]
        print(f"\ntrials:           {health['n_trials']}")
        print(f"unique positions: P1 {u[0]}, P2 {u[1]}")
        print(f"wall triggers:    P1 {t[0]}, P2 {t[1]}")
        print(f"step_penalty exceptions in Unity log: {health['n_exceptions']}")

    if min(health["uniq_positions"].values()) <= 1:
        raise RuntimeError("AN AGENT IS FROZEN — do not launch the round-robin")
    if health["n_exceptions"]:
        raise RuntimeError(f"Unity is still throwing on step_penalty "
                           f"({health['n_exceptions']}) — do not launch the round-robin")
    if verbose:
        print("\nSMOKE TEST PASSED — safe to run the full round-robin.")
    return health


# ---------------------------------------------------------------------------
# Reruns
# ---------------------------------------------------------------------------

def rerun_matchups(pairs, models_root, octagon_dir, unity_env_path, out_root, episodes,
                   trial_seq, seed=17, timeout_s=5000, base_port=10500, port_step=20,
                   archive_existing=True, verbose=True, **matchup_kwargs):
    """
    Rerun specific matchups that failed or came out short, without redoing the batch.

    Takes ordered (P1, P2) run_id pairs — the same ids that make up the `<P1>__vs__<P2>`
    dir name — and the batch config; returns {tournament_id: log path or None} and prints
    a health line per rerun. Everything else matches the round-robin, so the reruns are
    drop-in replacements.

    Actions are sampled, so changing `seed` re-rolls the trajectories: that is the lever
    for a matchup that *stalled* (both models wandering without triggering a wall) rather
    than crashed. Raising timeout_s alone mostly just buys a longer stall.

    With archive_existing, the previous output dir is moved to `out_root/../superseded/`
    first, so the rerun's JSON is the only one left in the matchup dir.
    """
    models_root, out_root = Path(models_root), Path(out_root)
    if not pairs:
        print("no pairs to rerun")
        return {}
    unknown = [m for pair in pairs for m in pair if not (models_root / m).exists()]
    if unknown:
        raise FileNotFoundError(f"unknown run_id(s) in pairs: {sorted(set(unknown))}")

    config_yaml = write_tournament_config_yaml(
        template_run_dir=models_root / pairs[0][0],
        out_yaml=out_root / "tournament_config.yaml",
    )

    logs = {}
    for i, (a, b) in enumerate(pairs):
        tournament_id = f"{a}__vs__{b}"
        out_path = out_root / tournament_id

        if archive_existing and out_path.exists():
            archive_to = (out_root.parent / "superseded"
                          / f"{tournament_id}__{time.strftime('%Y%m%d_%H%M%S')}")
            archive_to.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(out_path), str(archive_to))
            print(f"archived previous output -> {archive_to}")

        print(f"\n=== [{i+1}/{len(pairs)}] Rerun: {a} (P1) vs {b} (P2) ===")
        try:
            logs[tournament_id] = run_matchup(
                model_a_run_dir=models_root / a,
                model_b_run_dir=models_root / b,
                octagon_dir=octagon_dir,
                unity_env_path=unity_env_path,
                out_path=out_path,
                episodes=episodes,
                config_yaml=config_yaml,
                tournament_id=tournament_id,
                base_port=base_port + port_step * i,
                timeout_s=timeout_s,
                seed=seed,
                trial_seq=trial_seq,
                **matchup_kwargs,
            )
            print(f"=== Finished: {tournament_id} -> {logs[tournament_id]} ===")
        except Exception as e:
            logs[tournament_id] = None
            print(f"=== FAILED: {tournament_id} — {type(e).__name__}: {e} ===")

    if verbose:
        print()
        for tid in logs:
            health = matchup_health(out_root / tid, expected_trials=episodes,
                                    octagon_dir=octagon_dir, tournament_id=tid)
            print(format_health(tid, health))
    return logs
