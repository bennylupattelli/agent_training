"""
Cross-model inference tournament for Octagon.

Runs two trained models against each other in one inference session and saves a
behavioural-log JSON per matchup (the same contract octagon_analysis consumes).

How it works (see also the project memory notes):
  - The tournament Unity build (TournamentOctagonStage) has two agents with the
    DISTINCT behaviour names `CompetitiveAgent1` (tag PlayerAgent, log key "0")
    and `CompetitiveAgent2` (tag OpponentAgent, log key "1"), both BehaviorType
    Default so their policies run in Python via mlagents-learn.
  - mlagents loads policies keyed by behaviour name from
    results/<run-id>/<behaviour>/checkpoint.pt. To drive the two agents with two
    DIFFERENT trained models we assemble a synthetic results dir per matchup:
        results/<tournament_id>/CompetitiveAgent1/checkpoint.pt   <- model A
        results/<tournament_id>/CompetitiveAgent2/checkpoint.pt   <- model B
        results/<tournament_id>/configuration.yaml                <- both behaviours
    then run `mlagents-learn ... --resume --inference` (actions are SAMPLED, not
    greedy — see the deterministic note on run_matchup/run_tournament).

Reuses launch_inference_sim / run_eval from trainer_and_simulator_functions.py.
REMEMBER to activate the conda env with mlagents before running.
"""

from __future__ import annotations

import itertools
import shutil
from copy import deepcopy
from pathlib import Path

from ruamel.yaml import YAML

from trainer_and_simulator_functions import launch_inference_sim

yaml = YAML()
yaml.preserve_quotes = True

# Behaviour names baked into the tournament build (must match the scene exactly).
BEHAVIOUR_P1 = "CompetitiveAgent1"  # GameObject CompetitiveAgent1, tag PlayerAgent  -> log key "0"
BEHAVIOUR_P2 = "CompetitiveAgent2"  # GameObject CompetitiveAgent2, tag OpponentAgent -> log key "1"

# Subdir under <octagon_dir>/results/ where synthetic matchup run dirs are written.
TOURNAMENT_SUBDIR = "tournament"


def find_checkpoint(model_run_dir: Path) -> tuple[Path, str]:
    """
    Locate the trained policy checkpoint for one model run.

    Returns (checkpoint_pt_path, behaviour_name) where behaviour_name is the name
    of the subfolder that held it (e.g. "OctagonAgentSocial"). A standard
    mlagents run looks like:
        <model_run_dir>/<behaviour>/checkpoint.pt
        <model_run_dir>/configuration.yaml
    """
    model_run_dir = Path(model_run_dir)
    candidates = sorted(model_run_dir.glob("*/checkpoint.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No <behaviour>/checkpoint.pt found under {model_run_dir}"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Expected exactly one behaviour folder with checkpoint.pt under "
            f"{model_run_dir}, found {[c.parent.name for c in candidates]}"
        )
    ckpt = candidates[0]
    return ckpt, ckpt.parent.name


def _behaviour_block(config_path: Path, behaviour_name: str) -> dict:
    """Read a run's resolved configuration.yaml and return a deep copy of one
    behaviour block, with self_play disabled (inference uses a plain policy)."""
    config_path = Path(config_path)
    with config_path.open("r") as f:
        cfg = yaml.load(f)
    behaviours = cfg.get("behaviors")
    if not behaviours or behaviour_name not in behaviours:
        raise KeyError(
            f"Behaviour '{behaviour_name}' not in {config_path} "
            f"(found {list(behaviours) if behaviours else None})"
        )
    block = deepcopy(behaviours[behaviour_name])
    # Self-play needs >=2 agents of the SAME behaviour for ghosting; in the
    # tournament each behaviour drives one agent, so disable it for inference.
    if "self_play" in block:
        block["self_play"] = None
    return block


def _environment_parameters(config_path: Path) -> dict:
    """Return a deep copy of a run's environment_parameters block (e.g. step_penalty).

    These MUST be carried into the tournament config. At inference the mlagents
    communicator is on, so the build's isTraining is true and OnActionReceived reads
    step_penalty via Academy.EnvironmentParameters; if it is missing the build throws
    every action, which freezes the opponent agent. Copying the block over stops that.

    Raises if the block is absent rather than returning None: writing
    `environment_parameters:` (null) reproduces exactly the silent failure this
    guards against — the whole 260715 batch ran to completion, full-length JSONs
    and all, with the opponent frozen at its spawn and P1 winning 350/350.
    """
    config_path = Path(config_path)
    with config_path.open("r") as f:
        cfg = yaml.load(f)
    env = cfg.get("environment_parameters")
    if env is None:
        raise KeyError(
            f"No 'environment_parameters' in {config_path}. The build reads step_penalty "
            "from it on every action at inference (isTraining = IsCommunicatorOn) and throws "
            "without it, which freezes the opponent agent and invalidates the matchup."
        )
    return deepcopy(env)


def assemble_matchup_run_dir(
    model_a_run_dir: Path,
    model_b_run_dir: Path,
    octagon_dir: Path,
    tournament_id: str,
    behaviour_p1: str = BEHAVIOUR_P1,
    behaviour_p2: str = BEHAVIOUR_P2,
) -> str:
    """
    Build the synthetic results dir for one matchup and return the train_run_id
    to pass to mlagents-learn (relative to <octagon_dir>/results/).

    model_a -> behaviour_p1 (PlayerAgent),  model_b -> behaviour_p2 (OpponentAgent).
    The architecture for both behaviours is taken from model A's configuration.yaml
    (all tournament entrants share the same SocialConfig architecture); it must
    match the checkpoints or the weights will not load.
    """
    octagon_dir = Path(octagon_dir)
    model_a_run_dir = Path(model_a_run_dir)
    model_b_run_dir = Path(model_b_run_dir)

    ckpt_a, src_behaviour_a = find_checkpoint(model_a_run_dir)
    ckpt_b, _ = find_checkpoint(model_b_run_dir)

    train_run_id = f"{TOURNAMENT_SUBDIR}/{tournament_id}"
    run_dir = octagon_dir / "results" / TOURNAMENT_SUBDIR / tournament_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy each model's checkpoint into the matching behaviour folder.
    for behaviour, ckpt in ((behaviour_p1, ckpt_a), (behaviour_p2, ckpt_b)):
        bdir = run_dir / behaviour
        bdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ckpt, bdir / "checkpoint.pt")

    # Build a 2-behaviour configuration.yaml using model A's architecture for both.
    template = _behaviour_block(model_a_run_dir / "configuration.yaml", src_behaviour_a)
    cfg = {
        "default_settings": None,
        "behaviors": {
            behaviour_p1: deepcopy(template),
            behaviour_p2: deepcopy(template),
        },
        # carry step_penalty etc. so the build does not throw on every action at inference
        "environment_parameters": _environment_parameters(model_a_run_dir / "configuration.yaml"),
    }
    with (run_dir / "configuration.yaml").open("w") as f:
        yaml.dump(cfg, f)

    return train_run_id


def write_tournament_config_yaml(
    template_run_dir: Path,
    out_yaml: Path,
    behaviour_p1: str = BEHAVIOUR_P1,
    behaviour_p2: str = BEHAVIOUR_P2,
) -> Path:
    """
    Write the CLI config yaml passed to mlagents-learn (two behaviours, plain
    PPO, self_play disabled), derived from template_run_dir/configuration.yaml.
    Kept separate from the per-run configuration.yaml so one yaml can be reused
    across every matchup in a batch.
    """
    template_run_dir = Path(template_run_dir)
    out_yaml = Path(out_yaml)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)

    _, src_behaviour = find_checkpoint(template_run_dir)
    template = _behaviour_block(template_run_dir / "configuration.yaml", src_behaviour)
    cfg = {
        "default_settings": None,
        "behaviors": {
            behaviour_p1: deepcopy(template),
            behaviour_p2: deepcopy(template),
        },
        # carry step_penalty etc. so the build does not throw on every action at inference
        "environment_parameters": _environment_parameters(template_run_dir / "configuration.yaml"),
    }
    with out_yaml.open("w") as f:
        yaml.dump(cfg, f)
    return out_yaml


def run_matchup(
    model_a_run_dir: Path,
    model_b_run_dir: Path,
    octagon_dir: Path,
    unity_env_path: Path,
    out_path: Path,
    episodes: int,
    config_yaml: Path,
    tournament_id: str | None = None,
    base_port: int = 5015,
    timeout_s: int = 5000,
    seed: int | None = 17,
    # Actions are SAMPLED from the categorical policy (not greedy argmax) so play
    # matches how the models behaved during training. Trade-off: matchups are not
    # bit-reproducible run-to-run — use enough episodes to average over sampling.
    deterministic: bool = False,
    # Predetermined trial-sequence JSON (trial_sequences/generate_trial_sequence.py).
    # When given, every matchup replays the SAME string of trials and per-trial
    # timing, so trial content is pinned across models (requires a build compiled
    # with ScriptedTrialSequence.cs + timing keys). None => build draws trials
    # randomly from the GeneralGlobals probabilities.
    trial_seq: Path | str | None = None,
    behaviour_p1: str = BEHAVIOUR_P1,
    behaviour_p2: str = BEHAVIOUR_P2,
) -> Path:
    """
    Run a single two-model matchup in inference and produce one session JSON in
    out_path. This is the easy "two models against each other" entry point.

    model_a drives CompetitiveAgent1 (PlayerAgent / log key "0");
    model_b drives CompetitiveAgent2 (OpponentAgent / log key "1").
    """
    model_a_run_dir = Path(model_a_run_dir)
    model_b_run_dir = Path(model_b_run_dir)
    if tournament_id is None:
        tournament_id = f"{model_a_run_dir.name}__vs__{model_b_run_dir.name}"

    train_run_id = assemble_matchup_run_dir(
        model_a_run_dir=model_a_run_dir,
        model_b_run_dir=model_b_run_dir,
        octagon_dir=octagon_dir,
        tournament_id=tournament_id,
        behaviour_p1=behaviour_p1,
        behaviour_p2=behaviour_p2,
    )

    return launch_inference_sim(
        run_dir=octagon_dir,
        unity_env_path=unity_env_path,
        patched_yaml_path=config_yaml,
        train_run_id=train_run_id,
        out_path=out_path,
        episodes=episodes,
        base_port=base_port,
        timeout_s=timeout_s,
        seed=seed,
        deterministic=deterministic,
        trial_seq=trial_seq,
    )


def run_tournament(
    models_root: Path,
    octagon_dir: Path,
    unity_env_path: Path,
    out_root: Path,
    episodes: int,
    run_ids: list[str] | None = None,
    include_self_matchups: bool = True,
    # Also run each distinct pair in BOTH seat orders (A as P1 vs B, and B as P1 vs A).
    # Off by default since P1/P2 is symmetric; enable to double the data per pair and to
    # empirically confirm the seat has no effect by comparing a pair against its mirror.
    # Self-matchups are never duplicated.
    include_mirrors: bool = False,
    base_port: int = 5015,
    port_step: int = 20,
    timeout_s: int = 5000,
    seed: int | None = 17,
    # Sample actions (not greedy) to preserve training-time behaviour; see run_matchup.
    deterministic: bool = False,
    # Predetermined trial-sequence JSON replayed identically for every matchup so
    # trial content/timing is pinned across models; None => random trials. Keep the
    # sequence LONGER than `episodes` — the build caps the run at sequence length.
    trial_seq: Path | str | None = None,
    behaviour_p1: str = BEHAVIOUR_P1,
    behaviour_p2: str = BEHAVIOUR_P2,
):
    """
    Round-robin tournament over the models under models_root/<run_id>.

    Matchups: every unordered pair, INCLUDING self-matchups (toggle with
    include_self_matchups). By default NO mirrors — P1/P2 are symmetric, so each pair
    runs once. Set include_mirrors=True to also run each distinct pair in the swapped
    seat order (both A__vs__B and B__vs__A); this doubles the data per pair and lets you
    check the seat has no effect. One JSON per matchup lands under out_root/<A>__vs__<B>/.
    """
    models_root = Path(models_root)
    out_root = Path(out_root)

    if run_ids is None:
        run_ids = sorted(
            d.name for d in models_root.iterdir()
            if d.is_dir() and any(d.glob("*/checkpoint.pt"))
        )
    if len(run_ids) < 1:
        raise FileNotFoundError(f"No model run dirs with checkpoints under {models_root}")

    matchups: list[tuple[str, str]] = []
    for a, b in itertools.combinations_with_replacement(run_ids, 2):
        if a == b:
            if include_self_matchups:
                matchups.append((a, b))
        else:
            matchups.append((a, b))
            if include_mirrors:
                matchups.append((b, a))   # seat-swapped mirror, kept adjacent to its pair

    # One shared CLI config (architecture is identical across entrants).
    config_yaml = write_tournament_config_yaml(
        template_run_dir=models_root / run_ids[0],
        out_yaml=out_root / "tournament_config.yaml",
        behaviour_p1=behaviour_p1,
        behaviour_p2=behaviour_p2,
    )

    print(f"Running {len(matchups)} matchups over {len(run_ids)} models: {run_ids}")

    logs: dict[str, Path | None] = {}
    for i, (a, b) in enumerate(matchups):
        tournament_id = f"{a}__vs__{b}"
        print(f"\n=== [{i+1}/{len(matchups)}] Matchup: {a} (P1) vs {b} (P2) ===")
        try:
            log = run_matchup(
                model_a_run_dir=models_root / a,
                model_b_run_dir=models_root / b,
                octagon_dir=octagon_dir,
                unity_env_path=unity_env_path,
                out_path=out_root / tournament_id,
                episodes=episodes,
                config_yaml=config_yaml,
                tournament_id=tournament_id,
                base_port=base_port + port_step * i,
                timeout_s=timeout_s,
                seed=seed,
                deterministic=deterministic,
                trial_seq=trial_seq,
                behaviour_p1=behaviour_p1,
                behaviour_p2=behaviour_p2,
            )
            logs[tournament_id] = log
            print(f"=== Finished: {tournament_id} -> {log} ===")
        except Exception as e:
            print(f"=== FAILED: {tournament_id} — {type(e).__name__}: {e} ===")
            logs[tournament_id] = None

    return logs
