# Split action costs: biasing agents towards central or peripheral play

Started 2026-10-20. Cost design decided 2026-10-28, and the first batch (261029, solo) started
2026-10-29. This README covers the aim, the cost design and the reasoning behind it, how it was
built, and how to analyse the results.

> **REVISIT WITH FRESH EYES.** The cost design in section 4 was settled in one long discussion that
> changed its mind twice (from budget-matched costs to fixed `step_penalty`, then back, then fixed
> again). Before the social batch, or before writing anything up, re-check the logic of section 4
> from scratch: whether `step_penalty` should be fixed, whether the bounds (section 4.3) hold up, and
> whether per-step costs are the right unit for comparing translation with rotation. The points most
> worth challenging are marked **(check)**.

## 1. Aim and hypothesis

Some human players play centrally and some peripherally. The aim is a family of agents that covers
both, made by changing what actions cost the agent rather than the task. Each agent then serves as a
model of a player type.

**Hypothesis.** A high turning cost pushes agents to the periphery. A low turning cost, perhaps with a
low translation cost as well, lets them play centrally.

For agents the costs are reward penalties on actions. For humans they are abstract (effort,
attention, motor cost) and cannot be measured. So a cost found to reproduce a player's behaviour is
"the RL cost under which an agent plays like this player", not a measurement of what the player
experiences.

**What the existing models already suggest.** In 260929, peripheral models turned 728 degrees per ITI
against 1489 for central models, and started a trial with a wall in view more often (0.843 against
0.691). Central play needs scanning and peripheral play does not, so a turning cost should hurt
central play more. The same peripheral models also travelled less in the ITI (13.5 against 43.8
units), so **a translation cost may push agents to the periphery too**. The stylised game model
agrees (a rest-free movement cost of about 0.003 at gamma 0.998 moved the equilibrium to the wall; see
`training_peripheral_agents.md`). The central corner of the design is therefore probably low on both
costs, not just low on turning.

## 2. What already exists

Benny (Benedetta Gencarelli) built split-cost tooling in June and July 2026:

- **Unity:** branch `origin/split-cost_raycast-agents`, commit `c85469d`. It replaces `step_penalty`
  with `translation_cost` and `turning_cost`, applied separately (both if the agent moves and turns in
  the same step, none if it rests). It branched from a June version of `OctagonAgent.cs`, so it
  predates the 261014 raycast changes and later edits.
- **agent_training (main):** `parameter_sampling.sample_split_cost` (grid, uniform or log-uniform
  sampling of the two costs), `config_file_patching.patch_agents_yaml` (writes both costs into the
  config), `full_simulator_functions.split_cost_simulator_two_values` (trains N models per cost pair
  with a run manifest), and `configs/SoloSplitCostConfig.yaml` (a solo config with empty cost slots).
  These were written for solo agents and simulation-based inference, and draw random seeds.

**This is kept for reference only.** The plan is a new, simple implementation on each side (Unity,
training, analysis) built on the current code, rather than adapting this tooling. See section 5.

## 3. How costs worked before 261029

Up to the 261027 build, `OctagonAgent.cs` charged `step_penalty` on every step during training. The
clause that would skip it when resting had `&& false`, so resting was charged too. So this was a time
cost, not an effort cost: **resting cost exactly as much as moving.**

History of the value (from git, `OctagonAgent.cs`):

| commits | date | step penalty |
| --- | --- | --- |
| `33a38c4`, `6f49833` | to 2026-02-10 | `AddReward(-0.0001f)`, hard-coded |
| `ef9cea5` onward | from 2026-02-19 | `AddReward(-1e-3f)`, later read from `step_penalty` in the YAML |

Scale, from the 261020 TensorBoard logs (runs 0030-0040, at 4e6 steps):

| quantity | value |
| --- | --- |
| decision interval | 0.02 s (fixed timestep 0.02, `DecisionPeriod` 1) |
| one translation step | 0.2 units (10 units/s; forward + strafe is normalised, so also 0.2) |
| one rotation step | 7.2 deg (360 deg/s) |
| episode length | 440-456 steps (~9 s) |
| time cost per episode | ~0.45 |
| mean cumulative reward per episode | -0.04 to +0.08 |
| fixed delays per episode | 2 s trial end + U(2,5) s ITI + U(0.5,1.5) s trial start, mean ~325 steps |
| episode cap | `MaxStep` 1000 (scene override on both agents in SoloOctagonStage), so a timeout costs 1.0 |

Useful distances in steps: centre to wall (~16.75 units) is 84 steps, wall to opposite wall (~33.5
units) is 168, a 180 deg turn is 25 and a full 360 deg scan is 50.

The 261027 builds also changed the trial delays to the physics clock (commit `7533b9a`), so episodes
in 261027-and-later builds are ~0.5 s shorter in sped-up training than in 261014. 261020 is
therefore not an exact baseline for the split-cost batches, which have their own (0, 0) cell.

## 4. Cost design (decided 2026-10-28)

### 4.1 Three costs, `step_penalty` held fixed

Each step falls into one of four classes:

| step | cost |
| --- | --- |
| rest | `step_penalty` |
| translate only (move and/or strafe) | `step_penalty` + `translation_cost` |
| rotate only | `step_penalty` + `rotation_cost` |
| both | `step_penalty` + `translation_cost` + `rotation_cost` |

The two kinds of cost do different jobs:

- **`step_penalty` is a time cost.** It applies to all four classes equally, so it cannot make
  resting cheaper than moving. It sets urgency in the race and the cost of a timeout (1000 x 0.001 =
  1.0). Together with gamma 0.99 it is the recipe that gave a ~50/50 central/peripheral split in
  260723 (25 central, 24 peripheral, 3 intermediate, 3 unassigned by GMM in
  `model_characterisation_demo260723.ipynb`).
- **The action costs are effort costs.** The cost differences between the four classes depend only on
  `translation_cost` and `rotation_cost`.

**Decision: `step_penalty` stays at 0.001 in every cell, and the action costs are added on top.** The
(0, 0) cell is then the 50/50 recipe, and the only thing that changes between cells is effort.
Resting does not become more attractive in absolute terms: a timeout still costs 1.0. Reaching a far
wall does become relatively more expensive, which is the manipulation.

### 4.2 Why not keep the total cost the same (budget matching)

The first proposal set `step_penalty = 0.001 - translation_cost - rotation_cost`, to keep the "total"
at the old 0.001. This was rejected:

- **The total is only defined for one behaviour.** It holds for a step that both translates and
  rotates. Every other class gets cheaper than before. With translation at 8e-4 and rotation at 0,
  resting and rotating on the spot would each cost 2e-4, five times cheaper than now.
- **It changes two things at once.** Lowering `step_penalty` changes urgency and the timeout cost as
  well as effort, so the cells would differ in time cost as well.
- **Most of the time cost is a near-constant offset.** It falls in the ITI, which the agent cannot
  shorten, so matching the total buys little.

The cost of not matching is that the total goes up for agents that act. Agents have learned at a
total of 1e-3 since 2026-02-19 (and at 1e-4 before), but there is **no direct comparison of learning
before and after that change**. Other settings (`sequence_length`, `memory_size`) changed around
the same commit. So the record shows agents learn at 1e-3, not that a 10x rise is harmless.
**(check)**

### 4.3 Values and bounds

Pilot levels, each cost on its own with the other at 0: **1e-4, 3e-4, 1e-3, 3e-3**, plus (0, 0).

- **Lower bound ~1e-4.** The RND (curiosity) reward pays for reaching new states, which in effect
  rewards moving. Per step in 261020 it is ~5e-3 at the start of training, ~8e-5 at 1M steps and
  ~2e-5 at 4M. An action cost below ~1e-4 is no bigger than the curiosity bonus around 1M steps,
  when strategies seem to form, so it is probably washed out. It is also only a tenth of the time
  cost. These per-step RND figures average over all steps, so treat them as order of magnitude only.
  **(check: when do strategies actually form?)**
- **Upper bound 3e-3.** The pilot should go just past where the cost breaks good play. The trip from
  a wall to the opposite wall (168 steps) costs ~0.17 at 1e-3 and ~0.5 at 3e-3, nearly the whole 0.6
  gap between High and Low. So High choices on 180 deg trials should start to fail at 3e-3.
- **3e-3 has never been trained.** A translating agent then pays 4e-3 per step, 4x beyond any total
  agents have been trained at (1e-4 to 2e-3 is within 2x of a tested total). A failure at 3e-3 may be
  a learning failure rather than a strategy effect, and the analysis must tell the two apart (see
  competence in section 6). If learning fails, the parameters are read every step, so an ML-Agents
  curriculum can ramp the costs up.
- **Rotation bites less per task at the same per-step value.** A full scan is 50 steps (0.15 at
  3e-3), against 84 steps for a centre return. The per-step values are kept equal across the two
  costs, so they match per second of effort. If rotation shows no effect at 3e-3, add 1e-2.
  **(check: per step, per second, or per task?)**

### 4.4 Discounting and what the agent knows

The agent learns nothing about the next trial until the walls light at slice onset. Its waiting
position is a bet on the expected value of that position over all wall layouts, chosen in the ITI.
No cue marks the end of the ITI, so the agent must be in position by the earliest possible onset
(2 + 2 + 0.5 s after the previous trial ends, ~225 steps). Onset then comes 225-475 steps after the
trial ends.

So the trade-off for waiting position is the travel cost, paid early and barely discounted (84 steps
x cost for a centre return: 0.008 at 1e-4, 0.025 at 3e-4, 0.084 at 1e-3, 0.25 at 3e-3), against the
centre's advantage in expected discounted reward at onset, further discounted over the wait. There
is no good measure of that advantage: `score_rate` is score per second of trial time (see
`model_characterisation.py`), and `mean_score` comes from self-play. **(check)**

### 4.5 Hypotheses

The trial ends with the agent in a wall alcove, so doing nothing leaves it peripheral. A central agent
has to pay ~84 translation steps to go back to the centre every ITI. So **translation cost is
expected to be the stronger lever on waiting position**, and rotation cost to change mainly what
agents do once there.

- **Rotation cost and the ITI.** Turning in the ITI buys no information, since no walls are lit. A
  rotation cost should stop ITI spinning in every cell. It affects position only through the
  scanning a central agent needs after onset, which falls in the response phase and is discounted
  less.
- **Rotation workarounds.** With low translation and high rotation cost, an agent can still go back
  to the centre by strafing or moving backwards without turning. Expect only a mild shift to the
  periphery or intermediate radii. Track which action branches each cell uses.
- **High translation + high rotation: wall camper.** Stays at the last wall, turns ~180 deg once to
  face inward and rests a lot. Fast wins on adjacent walls, fewer High choices on 180 deg trials.
  Most likely to collapse to "always take the nearest wall".
- **High translation + low rotation: peripheral scanner.** About as peripheral, but scans from the
  wall, with more rotation, earlier detection and slightly more High choices than the camper.
- **Solo.** There is no race, so the centre's advantage only works through discounting. Solo agents
  may lean peripheral even at (0, 0), and the solo baseline in this build is unknown.
- **Social, later.** The opponent can end the trial, so letting it win a distant race becomes the
  cheaper option sooner at high costs. That is why the pilot runs in solo first.

## 5. Implementation (as built)

- **Unity** (Octagon `b0001a7`, branch `tom-dev_raycast-agents_split-cost`):
  `OctagonAgent.OnActionReceived` charges, during training only (`isTraining`):
  `step_penalty` every step, `translation_cost` once if moving and/or strafing, and `rotation_cost`
  if rotating. Costs are charged on the chosen action, not the displacement achieved, so pushing
  against a wall still costs. All three are read every step with `GetRequiredEnvironmentParameter`,
  which throws if a key is missing. The `&& false` / `throw` on rest is gone.
- **Build:** `builds/261029/build_261029_solo_01` (also `social_01` and `tournament_01`). See
  `builds/261029/notes.txt`. Every config used with it, including for inference, needs all three keys
  (set the action costs to 0.0 to reproduce the old reward). `tournament_func.py` copies the whole
  `environment_parameters` block, so it needs no change.
- **Configs:** template `Octagon/Assets/Scripts/MLConfigFiles/SoloSplitCostConfig261027.yaml` (the
  261020 hyperparameters, self-play removed, constant schedules). One config per cell in
  `configs/split_cost_261029/SoloSplitCost261029_<cell>.yaml`, which differ only in the costs.
- **Training:** the 261029 cell in `batch_inference_and_training_injectionless.ipynb`. Run ids are
  `261029_<cell>_000N` with 3 seeds per cell (`DEFAULT_TRAINING_SEEDS` 1-3, shared across cells). The
  extreme cells run first.
- **Smoke test (2026-10-29):** 40k steps of the `T3e-3` cell on the 261029 solo build. All three
  parameters were read, and the first mean reward was -3.68 against a predicted -3.67 for a random
  agent (1.0 time cost + 8/9 x 1000 x 0.003 translation cost over a timeout).
- **Not done:** per-trial cost totals are not logged in the JSON. They can be rebuilt from position
  and yaw logs.

## 6. Analysis

**Primary outcome: where models wait.** For each cell, the distribution of per-model mean onset
centrality, the peripheral share (cut at 0.5, with Clopper-Pearson intervals), and a GMM per cell.
Classify peripheral models by each batch's own GMM. Under the 0.5 cut the 260723 comparison
included intermediate models (see the 260929 notebook correction). Watch for a graded shift in
centrality as well as a change in share: costs may move the whole population rather than switch
seeds between two modes.

**Mechanism: what the cost changed first.** Per cell, `rot_total`, `prop_no_rotational`,
`iti_path_len`, `proportion_resting`, `frac_visible_at_onset` and `late_latency_to_wall_visible`.
The hypothesis predicts that a turning cost cuts scanning (`rot_total`), and that agents then stand
where walls are visible without turning, which is the periphery. If `rot_total` falls but centrality
does not move, the agents found another way to see walls cheaply.

**Look for workarounds.** Under a turning cost, agents can travel by strafing or moving backwards
without turning, since the 8-direction control scheme allows it. Humans can too, so this is not wrong
in itself, but check the share of steps that use each action branch per cell. The rays only look
forward (110 deg field of view), so turning mainly buys information, not travel.

**Competence.** `score_rate`, `p_high`, `latency_to_reach_wall` and `path_ratio` per cell. A cost
that produces peripheral agents by making them bad players is not a useful model. Expect `p_high` to
fall as costs rise, since High is often further away. That is a cost effect worth reporting, and
humans may show it too.

**Cross-cell tournaments.** Self-play under-penalises slow or passive play. Race models from
different cells against each other (peripheral from a high-turning-cost cell against central from
(0, 0)) to see whether the cost-shaped strategies hold up, and how they compare with human matchups.
`race_estimate` can screen pairs first.

**Mapping humans onto the grid.** Human fingerprints use the same metric definitions
(`octagon_analysis/READMEs/player_fingerprint_metrics.md`). For each human, find the cells whose
models are nearest in fingerprint space, or fit the costs by simulation-based inference, as the
original tooling intended. The questions that matter:

- **Identifiability.** If both costs push agents to the periphery, can the fingerprint tell them
  apart? A turning cost should cut rotation more than travel, and a translation cost the reverse. If
  the cells overlap in fingerprint space, a peripheral human cannot be assigned to one cost, and the
  claim has to be about total effort.
- **Human-agent differences that are not costs.** Mouse turning is near-instant and agents are
  capped at 360 deg/s. Humans get no per-step reward. Agents use 7 forward rays, humans a full
  screen. Compare on metrics these affect least (position at onset, ITI travel, resting), and treat
  latency comparisons with care.
- **Both strategy groups at once.** The claim is that one model family covers both player types. Show
  that central and peripheral humans map to different cells, not just that some cell matches each
  group's mean.

## 7. Decisions still open

- Re-check the section 4 logic with fresh eyes (see the flag at the top).
- Linear or constant schedules. Constant for the 261029 solo pilot. The 261020 characterisation
  (constant schedules, 261014 build) shows whether constant still gives the 50/50 mix that 260723
  (linear) gave.
- Levels for the combined grid, after the pilot. Choose the lowest cost with a detectable effect and
  the highest that still gives competent play.
- Seeds per cell for the full grid. 3 is enough for a pilot, but strategy is noisy across seeds, so a
  peripheral share needs more.
- Whether to add per-trial cost logging before the next build.
