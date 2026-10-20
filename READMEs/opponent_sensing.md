# Opponent sensing: what agents see of each other, and whether they use it

Written 2026-10-14. Covers how the two agents sense each other, how self-play training alternates the
learning agent between seats, a behavioural test of whether agents respond to where their opponent
stands, and a test of whether the trained networks use the opponent input at all.

The analysis lives in `octagon_analysis`: Stage 6 of
`demos/strategy_characterisation/models/competitive/model_characterisation_demo260929.ipynb`, with
code in `analysis/agent_raycast.py`, `analysis/opponent_response.py` and `analysis/policy_replay.py`
(tests in `tests/`).

## Summary

- **Before the 261014 builds, only one seat could sense its opponent.** The `PlayerAgent` (logged
  player 0) got the opponent on its own input channel. The `OpponentAgent` (logged player 1) only met
  the `PlayerAgent` as an untagged obstacle that blocked its rays.
- **Self-play switches the learning agent between seats every 750,000 steps.** For about half of
  every run, the agent learning was the seat without the opponent input.
- **Agents do not respond to where their opponent stands.** In cross-model tournaments, seeing the
  opponent near High at slice onset left the agent's heading unchanged (odds ratio 1.06, 95% interval
  0.65-1.71).
- **The trained networks have the opponent input but ignore it.** Replaying logged episodes with the
  opponent input removed changes the action probabilities by 1-3% as much as swapping the High and
  Low tags does.
- **From the 261014 builds, both seats pass through and report each other, walls hide the opponent,
  and the opponent value is proximity (1 - distance / 60) instead of distance.** The first builds made
  on 14 October predate the last two changes (see the build check below).

## How agents sense each other

The agents have no camera. `SelectivePassThroughRaycast` (`Octagon/Assets/Scripts/RaycastLogic/`)
casts 7 horizontal rays from the agent's position every physics step, at 0, ±18.3, ±36.7 and ±55
degrees from its heading (`maxRayDegrees: 55`, `raysPerDirection: 3`, `rayLength: 60`). Each ray gives
6 values to the network:

| value | meaning |
| --- | --- |
| distance / 60 | distance to the first object that is not passed through |
| HighWall, LowWall | 1 if that object is part of a lit wall (faces, alcove sides or alcove back) |
| HighWallTrigger, LowWallTrigger | 1 if it is a lit wall's trigger box (the back half of the alcove) |
| opponent | before 261014: distance / 60 to the nearest passed-through object anywhere along the ray, or 0. From 261014: proximity, 1 - distance / 60, or 0 (see below) |

That is 42 values, which matches the trained models' `obs_0` input (shape 42). A second input,
`obs_1`, has one value. `CollectObservations` adds nothing to it, so it is always 0.

The passed-through object is set by `passThroughTag = "OpponentAgent"`. The script takes the first
hit carrying that tag over the whole 60-unit ray, whether or not a wall is in the way. Before the
261014 builds this made the two seats unequal:

- `ArenaLogger.cs` logs the `PlayerAgent` as player 0 and the `OpponentAgent` as player 1.
- **Player 0** (`PlayerAgent`) passes through player 1 and reports its distance, even through walls.
- **Player 1** (`OpponentAgent`) has nothing to pass through, so its opponent channel is always 0.
  Player 0's body (tag `PlayerAgent`) is just the first hit of any ray that meets it. It reads as an
  untagged distance, the same as an unlit wall, and hides the walls behind it.

Agent bodies are the root `CharacterController` (radius 0.6). The character model is imported with
`addColliders: 0`, so nothing else on the opponent blocks or tags a ray. In the replayed episodes
below, player 0's opponent channel is non-zero on 19-32% of frames.

### Change in the 261014 builds

Two changes to `SelectivePassThroughRaycast.cs`, made on 2026-10-14.

1. **Both seats pass through each other.** `passThroughTag` became a list, `passThroughTags = {
   "OpponentAgent", "PlayerAgent" }` (commit `f524ff5`). Both seats now report their opponent's
   distance, and neither seat's rays stop at the other's body. An agent's rays start inside its own
   collider, so it still does not detect itself.
2. **Walls hide the opponent.** The opponent counts only if it is nearer than the first solid hit on
   that ray, the nearest hit that is neither an agent nor a trigger box (`occluderDistance`). An
   opponent behind a wall or an alcove side is no longer reported. One standing inside a trigger box
   is still seen, as trigger volumes do not occlude. This edit came after commit `f524ff5`.
3. **The opponent value is proximity, not distance.** It is `1 - d / 60` when an opponent is seen and
   0 when none is. With distance (`d / 60`), an opponent right next to the agent read about 0.01,
   almost the same as 'no opponent' (0). With proximity, real values fall between about 0.99 (touching)
   and 0.34 (across the arena, about 39 units), well clear of 0, and rise as the opponent comes closer.
   This is `1 - HitFraction` in ML-Agents' own `RayPerceptionSensor`, without its separate no-hit flag.

The observation size is unchanged (42), so older models still load, but the opponent channel now means
something different in both seats. Models trained before 261014 met it only in seat 0, through walls,
and as distance, so in a 261014 build they read it the wrong way round. They should not be compared
with 261014 models on opponent behaviour, and should not be run in 261014 builds for such comparisons.

**Build check.** The builds in `builds/261014` were remade at 13:41-13:42 on 14 October from commit
`5a661d6`, and contain all three changes. Decompiling `SelectivePassThroughRaycast` from both
`build_261014_social_01` and `build_261014_tournament_01` (identical in each) shows
`passThroughTags = { "OpponentAgent", "PlayerAgent" }`, the `isTrigger`-filtered `occluderDistance`, the
opponent hit limited to `h.distance < occluderDistance`, and the opponent value written as
`1f - distance / rayLength`. An earlier set of 261014 builds (11:53-11:54, from commit `f524ff5`, since
amended) had change 1 only, and has been overwritten. To check a build:

```bash
DOTNET_ROLL_FORWARD=Major ilspycmd -t SelectivePassThroughRaycast \
    <build>_Data/Managed/Assembly-CSharp.dll | grep -n "occluder\|isTrigger\|opponentHit ="
```

A build with the wall rule shows an `isTrigger` filter and a distance comparison in the `opponentHit`
line. A build with proximity shows `1f - ...distance / rayLength` where the opponent value is written.

The same edit gates the per-ray `Debug.Log` behind a serialized `debug` flag. It defaults to false
in the script, and neither the `PlayerAgent` prefab nor either training scene sets it, so the builds do
not log per ray. The prefab still holds the old `passThroughTag: OpponentAgent` value, which Unity ignores
now the field is a list. The list is not serialized anywhere, so the script's default list applies.

## Self-play switches the learning seat

Both agents run the `OctagonAgentSocial` behaviour. In `SoloOctagonStage` the `PlayerAgent` has
`TeamId: 0` (the prefab default), and the `OpponentAgent` ("PlayerAgent2") is overridden to
`TeamId: 1`.

The self-play settings are the same for 260715 and 260929: `team_change: 750000`, `swap_steps: 150000`,
`window: 8`, `play_against_latest_model_ratio: 0.5`. In ML-Agents' `ghost/trainer.py`, once the learner
has taken `team_change` steps since the last switch, `controller.change_training_team()` passes
learning to the next team in the queue (`ghost/controller.py`). With two teams, learning alternates
between team 0 and team 1 every 750,000 steps. The other team plays a frozen snapshot of the same
policy.

So a 4e6-step run has about 5 learning periods and a 7e6-step run about 9, roughly half in each seat.
Before 261014, the periods spent learning as the `OpponentAgent` had no opponent input at all. ML-Agents
logs the switch only at debug level, so the saved run files do not say which team learned first.

The policy is one network used in both seats. Before 261014, it learned from experience where the
opponent channel was zero about half the time.

## Behavioural test: does the opponent's position change the agent's choice?

The question is whether an agent gives up on High, or goes for it more, when it can see its opponent
near the High wall.

**Self-play cannot answer it.** A model's two agents start in the same alcove on 52% of trials and in
neighbouring alcoves on 32% (12.5% and 25% for random placement). The opponent's position cannot be
varied apart from the agent's own.

**Cross-model tournaments.** The 260715 and 260723 tournaments pit separately trained peripheral
models against each other (260715: 15, 26, 36; 260723: 1, 23, 33, 35, 51). There are 26 ordered
matchups, about 348 trials each, with self-matchups left out. Even here the agents start in the same
alcove on 26% of trials (12.5% when shuffled within a matchup). The link is the previous trial: its
winner stays in the alcove it just triggered on 95% of trials. The next trial's walls come from a
random sequence, so this is handled by comparing within the agent's own position.

**How it was measured.** Several obvious versions give false results, each caught by a control that
should show no effect (opponents near High that the agent never sensed).

| version | apparent effect (odds ratio) | control (never sensed) |
| --- | --- | --- |
| choice = wall a winner reached, or a called loser's choice; opponent sensed at any time | 1.92 | 0.15 |
| heading at 1.0 s for every agent; opponent sensed by 1.0 s | 1.42 | 0.20 |

Two things cause this. A winner's wall, or a loser whose choice can be called, depends on what both
agents did, and an opponent near High changes who wins. And sensing during the trial depends on where
the agent turns, so agents heading for High sweep up an opponent standing near High. The version
used avoids both:

- **Outcome:** the agent's heading at 1.0 s after slice onset (nearer the High or the Low alcove), for
  every trial lasting past 1.0 s (94%), winner or loser. The direction comes from the repo's
  `trajectory_direction.cosine_similarity_throughout_trajectory`. On winners it matches the wall reached
  77% of the time. That error is the same across opponent groups within a stratum, so it can shrink an effect
  but not create one.
- **Seen opponent:** a ray senses the opponent at slice onset, before any movement.
- **Comparison:** agents that saw their opponent against agents that did not, within the same position
  (alcoves to High and to Low) and the same view of the walls at onset (High in view, Low in view).
  Trials where the agents share an alcove, or either starts on a lit trigger, are left out. "Near High"
  means within 15 units of the High alcove centre, which takes in the neighbouring alcove.

**Result.**

| opponent at slice onset | trials | heading High at 1.0 s | stratified odds ratio | also adjusted for facing |
| --- | --- | --- | --- | --- |
| seen, near High | 94 | 0.691 | 1.12 (0.71-1.77) | 1.06 (0.65-1.71), p = 0.83 |
| seen, far from High | 488 | 0.621 | 0.82 (0.67-1.02) | 0.83 (0.67-1.02), p = 0.08 |
| not seen (reference) | 7461 | 0.663 | | |

Within strata the groups are matched on distance to both walls (within 0.6 units). The near group
still faces High 9.7 degrees more, which the regression adjusts for.

Seeing the opponent near High does not change where these agents head. Seeing it anywhere does not
reliably change it either. With 94 exposed trials, a moderate effect in either direction cannot be ruled out, only a
large one.

## Input test: do the trained networks use the opponent input?

The behavioural test cannot tell an agent that cannot see its opponent from one that sees it and does
not care. This test feeds the trained networks controlled inputs.

**Method.** `analysis/policy_replay.py` re-implements the exported policy in numpy:
concat(obs_0, obs_1) -> two 512-unit layers (swish) -> LSTM (128 units) -> three softmax heads (move,
strafe, rotate). For logged episodes it rebuilds each frame's 42 inputs from the positions and yaw:

- Memory resets at each trigger (`EndEpisode` in `OctagonWallTrigger.cs`), so an episode runs from the
  previous trial's trigger to this trial's trigger.
- Walls are washed and triggers switched off at the trigger, and both come back at slice onset. Wall
  tags and trigger boxes are therefore only present from slice onset.

It then replays each episode several times, changing one input each time:

- **Opponent removed.** Seat 0's opponent channel is zeroed. For seat 1, the opponent's body is made
  see-through.
- **Positive control.** The High and Low tags are swapped, which should change behaviour a lot.
- **Sanity control.** The ray order is reversed, which should make the replay worse at predicting what
  the agent actually did.

Change is measured as the total variation distance between the two sets of action probabilities
(0 = identical, 1 = completely different), averaged over the three action branches. It is taken from
the first frame the opponent changes the input.

**Result** (40 episodes per row; 260715 models from cross-model tournament matchups, 260929 model 32
from self-play):

| model | seat | opponent removed: change | High/Low swap: change | chosen action changes (opponent vs swap) | match to real actions (as logged / opponent removed / rays reversed) |
| --- | --- | --- | --- | --- | --- |
| 260715_0015 | 0 | 0.003 | 0.192 | 1.0% vs 57% | 0.731 / 0.731 / 0.438 |
| 260715_0026 | 0 | 0.001 | 0.123 | 0.3% vs 45% | 0.610 / 0.610 / 0.540 |
| 260715_0036 | 0 | 0.002 | 0.147 | 0.4% vs 54% | 0.569 / 0.569 / 0.396 |
| 260929_0032 | 0 | 0.007 | 0.286 | 2.2% vs 77% | 0.475 / 0.474 / 0.464 |
| 260715_0015 | 1 | 0.011 | 0.198 | 3.1% vs 59% | 0.729 / 0.729 / 0.445 |
| 260715_0026 | 1 | 0.011 | 0.112 | 3.1% vs 41% | 0.617 / 0.617 / 0.540 |
| 260715_0036 | 1 | 0.024 | 0.150 | 6.8% vs 54% | 0.561 / 0.559 / 0.398 |
| 260929_0032 | 1 | 0.027 | 0.279 | 8.8% vs 82% | 0.461 / 0.459 / 0.420 |

"Match to real actions" is the mean probability the network gives the action implied by the agent's
actual movement on the next frame.

- **Seat 0 has the opponent input and barely uses it.** Removing it moves the action probabilities
  1-3% as much as swapping the wall tags, and leaves the match to real actions unchanged.
- **Seat 1's effect is a little larger, but it is not a clean opponent measure.** Making the
  opponent see-through also changes which walls its rays reach.
- **The replay is trustworthy for the 260715 models.** It gives real actions well above chance, and
  reversing the rays drops the match (e.g. 0.731 to 0.438 for model 15). For 260929 model 32 the match
  is weaker and barely drops when the rays are reversed, so its rows deserve less weight.

So the agents could sense their opponent for half of training, and the finished policies do not use
it. The opponent input being zero for half the training experience may be part of the reason. That is
untested.

**Assumptions.**
- `obs_1` is 0.
- Both seats take a decision on every logged frame, including the ITI.
- The 260715 replay used the `.onnx` exported at the end of training. The tournament loaded
  `checkpoint.pt`, whose folder was modified three days later. These are believed to be the same final
  policy, but this has not been confirmed.

## What to check with 261014 builds

1. **Confirm the build has both changes** with the decompile check above.
2. **After training a 261014 batch,** run `policy_replay.opponent_input_check` for both seats with
   `**policy_replay.OBS_261014` (both seats, walls occlude, proximity). That asks whether full access during
   training leads the policy to use the opponent. With the same seeds and settings as 260929, the two
   batches differ only in what the opponent channel carries.
3. **Rerun the behavioural test** on a 261014 tournament with cross-model matchups. The previous-trial
   link between the agents' positions is part of the task, so the stratified comparison is still
   needed.

## Running the code

The weights are read from the exported `.onnx`, which needs the `onnx` package (in the `mlagents`
env). Everything else runs in the `octagon_analysis` env.

```python
# in the mlagents env
from analysis.policy_replay import export_weights
export_weights('<run>/OctagonAgentSocial.onnx', 'weights.npz')

# in the octagon_analysis env
import analysis.policy_replay as pr, analysis.agent_raycast as ar
eps = list(pr.episodes(ar.matchup_trial_list(tournament_dir, '<A>__vs__<B>')))
pr.opponent_input_check(pr.Policy('weights.npz'), eps[:40], seat=0, **pr.OBS_PRE_261014).mean()
# models trained in 261014 builds or later
pr.opponent_input_check(pr.Policy('weights.npz'), eps[:40], seat=1, **pr.OBS_261014).mean()
```

`opponent_input_check` returns one row per episode with the columns used in the table above. The
observation rules of each build generation are `policy_replay.OBS_PRE_261014` and
`policy_replay.OBS_261014`, which `build_obs` also takes. `agent_raycast.BUILD_PRE_261014` and
`agent_raycast.BUILD_261014` hold the sensing rules alone (seats and occlusion), for `opponent_sensed`.
