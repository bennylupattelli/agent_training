# Predetermined trial sequences

Reproducible strings of Octagon trials for inference. When a sequence file is
supplied, every model replays the **same** trials, so behaviour is comparable
across models (action selection is already reproducible; this makes the trials
reproducible too). Without a sequence file the Unity build generates trials
randomly, exactly as before.

## Generate a sequence

```bash
python trial_sequences/generate_trial_sequence.py            # 100k trials, seed 17
python trial_sequences/generate_trial_sequence.py -n 20000 -s 42 -o my_seq.json
```

Sampling matches `Globals/General.cs` and `OctagonArenaSettings`:
random anchor wall (= high wall), separation ∈ {1,2,4} weighted 50/25/25,
CW/CCW second wall, trial type `HighLow`. Edit the `generate_trials` defaults
(`trial_types`, `separations`, weights) if those game-side constants change.

## Use it in inference

Pass `trial_seq=` to `batch_inference(...)` (or `launch_inference_sim(...)`):

```python
batch_inference(..., episodes=5000,
                trial_seq="trial_sequences/trials_100000_seed17.json")
```

This forwards `--trial_seq <path>` to the Unity build via `--env-args`. The build
loads it in `ScriptedTrialSequence.cs` and **caps the run at the sequence length**:
inference stops once the trials are exhausted, so keep the sequence longer than
`episodes` (the default 100k covers typical runs). Omit `trial_seq` for random
trials.

## File format

```json
{
  "meta_seed": 17,
  "meta_numWalls": 8,
  "meta_generatedTrials": 100000,
  "trials": [
    {"trialType": "HighLow", "separation": 1, "direction": -1,
     "anchorWallID": 6, "highWallID": 6, "lowWallID": 5},
    ...
  ]
}
```

`highWallID`/`lowWallID` are the custom wall IDs the arena assigns directly
(anchor is always the high wall); `separation`/`direction`/`anchorWallID` are
included for inspection. Wall IDs are the sorted set `[1..8]`.
