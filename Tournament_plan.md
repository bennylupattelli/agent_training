# Tournament Plan — cross-model inference

Brief for building a **tournament** between trained Octagon models in this repo. The goal: run
each (current planned to be hand-chosen) model against every other (and itself), to compare strategies (e.g. "centre" vs "surround")
under competition, and produce per-matchup behavioural logs for `octagon_analysis`.


## The core problem

Standard inference here (`launch_inference_sim`) runs **one model** via
`mlagents-learn ... --run-id <id> --resume --inference --env <build> --env-args --sim_out --sim_eps`.
A tournament needs **two different models** (from two different runs) driving the two agents in one inference 
session.


## Reuse from this repo
Integrate into this repo so that I am reusing code effectively

## Output 
Octagon_analysis requires a .json file containing logs of agent behaviour over time. Just one .json file
should be fine for analysis.

## Tournament design decisions (already made)

- **Matchups:** TBD. However, no need for mirror matchups, as it doesn't matter which model is 'Player1'
and which is 'Player2'
- **Trial mix:** experimental **50/25/25** across 45/90/180° wall separations (set in Octagon's
  `GeneralGlobals.cs`); 
- **Reproducibility:** set deterministic inference so trial variation comes from the stochastic
  arena, not action sampling.

## Repos

- `/home/tom/Unity/Octagon` — the Unity environment + build.
- `/home/tom/repos/octagon_analysis` — consumes the simulation logs (the output contract above).
