"""
Pre-flight, health and rerun helpers for the cross-model tournament.

A matchup can exit cleanly, write a full-length JSON, and still be void. Before the
260729 fix, a missing environment_parameters block froze one agent and gave the other
every trial, and the exit status looked fine. So the checks here read behaviour rather
than exit codes:

  - both agents occupy more than one position (an agent that never moves is frozen),
  - both agents trigger some walls (one player winning everything is the same fault),
  - the trial count is close to the requested episode count (a short run stalled).

Per-batch config and results stay in the notebooks (tournament/run_tournament*.ipynb).
Checks read the session JSON by regex instead of going through octagon_analysis, so this
module imports cleanly in the mlagents env.
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

# How many trials short of the request before a run counts as short. Sampled actions cost
# a trial or two; a real stall (68 of ~350, in the 260715_0002 self-matchup) is far below.
TRIAL_SHORTFALL_TOLERANCE = 5

# Ports one inference matchup needs from its base port.
MATCHUP_PORT_SPAN = 4

# Base port for one-off matchups such as the smoke test. Not run_matchup's own default of
# 5015: training takes 5005 + 20*i with 16 envs each, so 5015 falls inside its first block.
# The round-robin uses 9100+ and reruns 10500+, so 11000 is clear of all three.
SMOKE_TEST_BASE_PORT = 11000


def ports_in_use(base_port, n_ports):
    """Return the bound TCP ports in [base_port, base_port + n_ports)."""
    import socket
    busy = []
    for port in range(int(base_port), int(base_port) + int(n_ports)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("localhost", port))
            except OSError:
                busy.append(port)
    return busy


def assert_ports_free(base_port, n_ports, what="this run"):
    """
    Raise if a port the run needs is already bound. Takes the range and a label.

    mlagents checks this itself, but only after loading the build and policies, and it
    reports a UnityWorkerInUseException about "worker number 0" without naming the port.
    Concurrent training is the usual cause: it takes 5005 + 20*i with 16 envs per model.
    """
    busy = ports_in_use(base_port, n_ports)
    if busy:
        raise RuntimeError(
            f"{what} wants ports {base_port}..{base_port + n_ports - 1}, but "
            f"{len(busy)} are already bound: {busy[:8]}{'...' if len(busy) > 8 else ''}. "
            "Concurrent training takes 5005 + 20*i (16 envs each); pick a base_port clear of "
            "it, or stop the other job."
        )


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

def preflight(models_root, unity_env_path, trial_seq, run_ids=None,
              include_mirrors=True, include_self_matchups=True,
              base_port=9100, port_step=20, verbose=True):
    """
    Check a batch can run, and return the number of matchups. Raises on anything wrong.

    Takes the batch config. The config probe runs from the caller rather than inside
    tournament_func, so it also catches a kernel holding pre-fix code, which an assertion
    inside that module could not.
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

    # Without environment_parameters the build throws on every action, one agent never gets
    # stepped, and the matchup is a walkover that still writes a healthy-looking JSON.
    with tempfile.TemporaryDirectory() as tmp:
        probe = write_tournament_config_yaml(models_root / run_ids[0], Path(tmp) / "probe.yaml")
        txt = probe.read_text()
    if "environment_parameters" not in txt or "step_penalty" not in txt:
        raise RuntimeError(
            "Generated config has no environment_parameters/step_penalty. The build would "
            "throw every action and freeze P2. The kernel is running pre-fix code: restart "
            "it and re-run from the top."
        )

    n_matchups = len(expected_matchups(run_ids, include_mirrors, include_self_matchups))

    # The round-robin walks base_port + port_step*i, so a collision at matchup 40 would kill
    # the batch 20 hours in. Check the whole span now.
    if base_port is not None:
        assert_ports_free(base_port, port_step * (n_matchups - 1) + MATCHUP_PORT_SPAN,
                          what=f"the {n_matchups}-matchup round-robin")

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

    Takes the matchup output dir, optionally the requested episode count (to flag short
    runs) and the Octagon dir plus tournament id (to count step_penalty exceptions in the
    Unity log). Keys: n_trials, uniq_positions, triggers, n_exceptions, flags, ok.
    """
    matchup_dir = Path(matchup_dir)
    log = newest_session_json(matchup_dir)
    if log is None:
        return dict(json=None, n_trials=0, uniq_positions={0: 0, 1: 0}, triggers={0: 0, 1: 0},
                    n_exceptions=None, flags=["no JSON written"], ok=False)

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
        flags.append(f"short (expected ~{expected_trials - 1}, got {n_trials})")
    if min(uniq.values()) <= 1:
        flags.append("agent frozen, matchup is void")
    if min(triggers.values()) == 0:
        flags.append("one player won every trial")
    if n_exceptions:
        flags.append(f"{n_exceptions} step_penalty exceptions in the Unity log")

    return dict(json=log, n_trials=n_trials, uniq_positions=uniq, triggers=triggers,
                n_exceptions=n_exceptions, flags=flags, ok=not flags)


def format_health(tournament_id, health):
    """One-line summary of a matchup_health dict, for printing."""
    if health["json"] is None:
        return f"{tournament_id}: no JSON written"
    u, t = health["uniq_positions"], health["triggers"]
    tail = "  OK" if health["ok"] else "  <-- " + "; ".join(health["flags"])
    return (f"{tournament_id}: {health['n_trials']} trials | unique positions "
            f"P1 {u[0]}, P2 {u[1]} | triggers P1 {t[0]}, P2 {t[1]}{tail}")


def check_batch(out_root, tournament_ids, expected_trials=None, octagon_dir=None,
                logs=None, verbose=True):
    """
    Validate a whole batch. Takes the batch output dir and the matchup ids to check
    (usually logs.keys()); returns the list of suspect ids.

    Passing logs (the dict run_tournament returned) also reports which matchups raised.
    Only suspect matchups are printed, so a clean batch prints two summary lines.
    """
    out_root = Path(out_root)
    tournament_ids = list(tournament_ids)

    if logs is not None:
        failed = [k for k, v in logs.items() if v is None]
        if verbose:
            print(f"{len(logs) - len(failed)}/{len(logs)} matchups ran without error")
            for k in failed:
                print("  raised:", k)

    suspect = []
    if verbose:
        print("\nValidity check (both agents move, both win some trials):")
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


def outstanding_matchups(out_root, run_ids, expected_trials=None, include_mirrors=True,
                         include_self_matchups=True, verbose=True):
    """
    Return the (P1, P2) pairs a batch still needs, ready to hand to rerun_matchups.

    Takes the batch output dir and the entrant list. A matchup counts as outstanding when
    its dir is missing, holds no JSON, has no DONE.txt, or fails matchup_health. Anything
    already complete and healthy is left alone.

    Derived from what is on disk rather than from a hand-written list, so it can be called
    again after an interruption and will return only what is still left.
    """
    out_root = Path(out_root)
    todo, done = [], 0
    for a, b in expected_matchups(run_ids, include_mirrors, include_self_matchups):
        tid = f"{a}__vs__{b}"
        d = out_root / tid
        if not d.exists():
            todo.append((a, b, "no dir"))
            continue
        health = matchup_health(d, expected_trials=expected_trials)
        if not (d / "DONE.txt").exists():
            todo.append((a, b, "no DONE.txt"))
        elif not health["ok"]:
            todo.append((a, b, "; ".join(health["flags"])))
        else:
            done += 1

    if verbose:
        total = done + len(todo)
        print(f"{done}/{total} matchups complete, {len(todo)} outstanding")
        for a, b, why in todo:
            print(f"  {a}__vs__{b}  ({why})")
    return [(a, b) for a, b, _ in todo]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test(model_a_run_dir, model_b_run_dir, octagon_dir, unity_env_path, out_root,
               trial_seq, episodes=8, config_template_run_dir=None,
               base_port=SMOKE_TEST_BASE_PORT, verbose=True, **matchup_kwargs):
    """
    Run one short matchup and verdict it, before committing to a full round-robin.

    Takes the two entrant run dirs and the batch config; writes into out_root/_smoke_test
    and returns the health dict. Raises if an agent is frozen or Unity is throwing on
    step_penalty, the two faults that void a batch without any visible error.
    """
    out_root = Path(out_root)
    assert_ports_free(base_port, MATCHUP_PORT_SPAN, what="the smoke test")
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

    # The run is short by design, so don't check the trial count. Only frozen and
    # one-sided matchups matter here.
    tid = f"{Path(model_a_run_dir).name}__vs__{Path(model_b_run_dir).name}"
    health = matchup_health(out_root / "_smoke_test", octagon_dir=octagon_dir, tournament_id=tid)
    if verbose:
        u, t = health["uniq_positions"], health["triggers"]
        print(f"\ntrials:           {health['n_trials']}")
        print(f"unique positions: P1 {u[0]}, P2 {u[1]}")
        print(f"wall triggers:    P1 {t[0]}, P2 {t[1]}")
        print(f"step_penalty exceptions in Unity log: {health['n_exceptions']}")

    if min(health["uniq_positions"].values()) <= 1:
        raise RuntimeError("An agent is frozen. Do not launch the round-robin.")
    if health["n_exceptions"]:
        raise RuntimeError(f"Unity is still throwing on step_penalty "
                           f"({health['n_exceptions']}). Do not launch the round-robin.")
    if verbose:
        print("\nSmoke test passed. Safe to run the full round-robin.")
    return health


# ---------------------------------------------------------------------------
# Reruns
# ---------------------------------------------------------------------------

def rerun_matchups(pairs, models_root, octagon_dir, unity_env_path, out_root, episodes,
                   trial_seq, seed=17, timeout_s=5000, base_port=10500, port_step=20,
                   archive_existing=True, verbose=True, **matchup_kwargs):
    """
    Rerun specific matchups that failed or came out short, without redoing the batch.

    Takes ordered (P1, P2) run_id pairs, the same ids that make up the <P1>__vs__<P2> dir
    name, and the batch config. Returns {tournament_id: log path or None} and prints a
    health line per rerun. Everything else matches the round-robin, so the reruns are
    drop-in replacements.

    Actions are sampled, so changing seed re-rolls the trajectories. That is the lever for
    a matchup that stalled, where both models wander without triggering a wall; raising
    timeout_s alone tends to buy a longer stall.

    With archive_existing, the previous output dir moves to out_root/../superseded/ first,
    leaving the rerun's JSON as the only one in the matchup dir.
    """
    models_root, out_root = Path(models_root), Path(out_root)
    if not pairs:
        print("no pairs to rerun")
        return {}
    unknown = [m for pair in pairs for m in pair if not (models_root / m).exists()]
    if unknown:
        raise FileNotFoundError(f"unknown run_id(s) in pairs: {sorted(set(unknown))}")

    assert_ports_free(base_port, port_step * (len(pairs) - 1) + MATCHUP_PORT_SPAN,
                      what=f"the {len(pairs)}-matchup rerun")

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
            print(f"=== Failed: {tournament_id}. {type(e).__name__}: {e} ===")

    if verbose:
        print()
        for tid in logs:
            health = matchup_health(out_root / tid, expected_trials=episodes,
                                    octagon_dir=octagon_dir, tournament_id=tid)
            print(format_health(tid, health))
    return logs
