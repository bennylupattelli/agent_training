#!/usr/bin/env python3
"""Generate a predetermined string of Octagon trials for reproducible inference.

The output JSON is consumed by the Unity build (Assets/Scripts/TrialLogic/
ScriptedTrialSequence.cs) via the `--trial_seq <path>` command-line argument.
When that argument is supplied the arena replays this exact sequence of trials
instead of sampling them at runtime, so different models see an identical string
of trials and results are directly comparable.

The sampling here mirrors OctagonArenaSettings.SelectNewWalls / SelectTrial /
ITI():
  * anchor wall        : uniform over the N walls (anchor == the high wall)
  * wall separation    : weighted choice over `separations` (default 1/2/4)
  * direction (CW/CCW) : uniform +/- (low wall sits `direction * sep` from anchor)
  * trial type         : weighted choice over `trial_types`
  * iti                : uniform over [ITIMin, ITIMax) seconds (default 2..5)
  * trialStartDelay    : uniform over [0.5, 1.5) seconds

The two timing fields are predetermined so inference is fully reproducible: with
a scripted sequence loaded the arena reads them from each trial instead of
drawing Random.Range at trial boundaries (OctagonArenaSettings.ITI()). They are
drawn AFTER the wall/type draws, so for a given seed the wall sequence is
byte-identical to sequences generated before timing fields existed.

Wall custom IDs are the sorted IDs [1..N] (see IdentityManager.ListCustomIDs),
so anchor index i maps to wall ID i+1, and the dependent wall wraps circularly.

Defaults reflect the statistics currently defined in Globals/General.cs:
  trial_types = {"HighLow": 1.0}
  separations = [1, 2, 4] with weights [50, 25, 25]
  num_walls   = 8
"""

import argparse
import json
from pathlib import Path

import numpy as np


def generate_trials(
    n_trials: int,
    seed: int,
    num_walls: int = 8,
    separations=(1, 2, 4),
    separation_weights=(50, 25, 25),
    trial_types=("HighLow",),
    trial_type_weights=(1.0,),
    iti_range=(2.0, 5.0),
    trial_start_delay_range=(0.5, 1.5),
):
    """Return a list of trial dicts matching the in-engine sampling statistics.

    iti_range / trial_start_delay_range mirror the in-engine draws in
    OctagonArenaSettings.ITI():
        iti             = Random.Range(General.ITIMin, General.ITIMax)  # 2..5 s
        trialStartDelay = Random.Range(0.5f, 1.5f)                       # 0.5..1.5 s
    Keep them in sync if those constants change.
    """
    rng = np.random.default_rng(seed)

    separations = np.asarray(separations, dtype=int)
    sep_p = np.asarray(separation_weights, dtype=float)
    sep_p = sep_p / sep_p.sum()

    tt_p = np.asarray(trial_type_weights, dtype=float)
    tt_p = tt_p / tt_p.sum()

    # Vectorised draws for speed and reproducibility.
    anchor_idx = rng.integers(0, num_walls, size=n_trials)                 # 0..N-1
    sep = rng.choice(separations, size=n_trials, p=sep_p)                  # 1/2/4
    direction = rng.choice((-1, 1), size=n_trials)                        # CW / CCW
    tt_idx = rng.choice(len(trial_types), size=n_trials, p=tt_p)

    # Timing draws come AFTER the wall/type draws so the wall sequence for a given
    # seed is unchanged from before these fields existed (see module docstring).
    iti = rng.uniform(iti_range[0], iti_range[1], size=n_trials)          # 2..5 s
    trial_start_delay = rng.uniform(                                      # 0.5..1.5 s
        trial_start_delay_range[0], trial_start_delay_range[1], size=n_trials)

    dependent_idx = (anchor_idx + direction * sep) % num_walls            # circular wrap

    trials = []
    for i in range(n_trials):
        anchor = int(anchor_idx[i])
        dep = int(dependent_idx[i])
        trials.append(
            {
                "trialType": trial_types[int(tt_idx[i])],
                "separation": int(sep[i]),
                "direction": int(direction[i]),
                "anchorWallID": anchor + 1,   # sorted IDs are 1..N
                "highWallID": anchor + 1,     # anchor is always the high wall
                "lowWallID": dep + 1,
                "iti": float(iti[i]),                       # seconds
                "trialStartDelay": float(trial_start_delay[i]),  # seconds
            }
        )
    return trials


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--n-trials", type=int, default=100_000,
                        help="Number of trials to generate (default: 100000).")
    parser.add_argument("-s", "--seed", type=int, default=17,
                        help="RNG seed (default: 17).")
    parser.add_argument("--num-walls", type=int, default=8,
                        help="Number of walls in the octagon (default: 8).")
    parser.add_argument("-o", "--out", type=Path, default=None,
                        help="Output JSON path (default: trials_<n>_seed<seed>.json "
                             "next to this script).")
    args = parser.parse_args()

    trials = generate_trials(args.n_trials, args.seed, num_walls=args.num_walls)

    out = args.out
    if out is None:
        out = Path(__file__).resolve().parent / f"trials_{args.n_trials}_seed{args.seed}.json"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta_seed": args.seed,
        "meta_numWalls": args.num_walls,
        "meta_generatedTrials": len(trials),
        "trials": trials,
    }
    with open(out, "w") as f:
        json.dump(payload, f)

    # Brief summary so the empirical distribution can be eyeballed.
    seps = np.array([t["separation"] for t in trials])
    print(f"Wrote {len(trials)} trials to {out}")
    for s in sorted(set(seps.tolist())):
        frac = (seps == s).mean()
        print(f"  separation {s}: {frac:6.2%}")


if __name__ == "__main__":
    main()
