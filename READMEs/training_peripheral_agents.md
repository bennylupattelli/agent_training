# Training peripheral agents

A starting point for training social models that adopt a peripheral strategy. Written after the
first five models of the 260914 batch all came out central. Covers why the batch went central,
what the game itself pays for, and which changes are most likely to produce peripheral models.

Positions are given both as a radius in Unity units (walls at 16.75, alcove backs at 19.747) and
as the centrality used in analysis, `1 - r / OCTAGON_ALCOVE_RADIUS`, where 1 is the centre and 0
is the alcove back. `split_strategies` in
[octagon_analysis/analysis/model_characterisation.py](/home/tom/repos/octagon_analysis/analysis/model_characterisation.py)
calls a model central above centrality 0.5, which is a radius of about 9.9.

---

## 1. Motivation

Human players show a range of strategies, including peripheral play. Agents that only ever play
centrally cannot serve as models of that behaviour. Two questions follow:

1. What can be changed in training to produce peripheral agents?
2. Is there any version of the game where peripheral play is actually the better way to score,
   given the wall separation probabilities, the movement and turning costs, and the ratio between
   the high and low rewards?

Central play may well be the best strategy under the current settings. The aim is not to overturn
that, but to find settings that make peripheral play competitive, so a peripheral population can
be trained without simply forcing it.

## 2. What the game pays

From the Unity source, for the build used by the 260914 batch:

| Fact | Value | Source |
|---|---|---|
| High / low reward | 1.0 / 0.4, loser gets 0 | `General.highScore`, `General.lowScore` in `Assets/Scripts/Utility/GeneralGlobals.cs` |
| Trial ends | at the first agent to enter either lit trigger | `HandleWallTrigger` in `Assets/Scripts/TrialLogic/OctagonWallTrigger.cs` |
| Wall separations | 1, 2, 4 with weights 50, 25, 25 | `General.wallSeparations*` |
| Episode | previous trial end, then 2 s, ITI U(2,5), trial start U(0.5,1.5), then the response | `ITI()` in `Assets/Scripts/TrialLogic/OctagonArenaSettings.cs` |
| Movement | 10 u/s, turning 360 deg/s, 50 decisions/s | `OctagonAgent.cs`, fixed timestep 0.02, `DecisionPeriod: 1` |
| Field of view | 110 deg (`maxRayDegrees: 55`) | `Assets/Prefabs/Player/PlayerAgent.prefab` |
| Step penalty | charged on every step, including rests | the `&& false` at `OctagonAgent.cs:529` |

Two consequences matter for positioning.

- The ITI sits at the start of the episode that ends with the reward, so where an agent waits is
  learned inside the same episode as the reward it leads to. The reward arrives about 325 steps
  after the agent repositions.
- With `&& false` in place there is no movement-specific cost. Resting and moving cost the same,
  so nothing rewards standing still. Removing `&& false` makes resting free, but only because that
  branch throws an exception rather than skipping the penalty cleanly.

## 3. The two batches

260723 produced roughly half central and half intermediate models. 260914 has produced central
models so far. Both were trained with the step penalty charged on every step, so the penalty type
is not the difference. Diffing the two saved `configuration.yaml` files gives the complete set of
differences:

| Setting | 260723 | 260914 | Affects training |
|---|---|---|---|
| extrinsic `gamma` | 0.99 | 0.998 | yes |
| `lambd` | 0.95 | 0.99 | yes |
| `time_horizon` | 128 | 512 | yes |
| `learning_rate_schedule` | linear | constant | yes |
| `epsilon_schedule` | linear | constant | yes |
| `max_steps` | 4e6 | 7e6 | yes |
| `even_checkpoints` | false | true | no |
| `checkpoint_interval` | 100000 | 175000 | no |
| `seed` | 17 | randomised per run | no |
| `env_path` | 260723 build | 260914 build | not verified |

Everything else is identical: `beta` 0.005, `beta_schedule` linear, `learning_rate` 1.5e-4,
`batch_size` 512, `buffer_size` 20480, `epsilon` 0.2, `num_epoch` 3, `sequence_length` 64,
`memory_size` 256, `hidden_units` 512, RND `gamma` 0.99 and `strength` 0.01, `step_penalty` 0.001,
and all four `self_play` settings.

### Why each change was made

The training-relevant changes came out of the 260723 TensorBoard logs, which showed two separate
problems: training stopped before it converged, and the discount horizon was far shorter than the
task.

- **`learning_rate_schedule` and `epsilon_schedule`, linear to constant.** Linear decay takes the
  learning rate to zero at `max_steps`. In 260723 run 0001 it was 1.0e-4 at 1M, 5.1e-5 at 2M and
  6.3e-7 at 3M, while ELO was still climbing over the same stretch: 1383 at 2.0M, 1415 at 2.5M,
  1424 at 3.0M. The runs were ended by the schedule, not by convergence. The 3e6 to 4e6 extension
  confirms it. Resuming keeps the global step count, so the schedule gave 1.9e-5 at 3.5M falling
  back to zero, ELO moved by about 10 points either way, and episode length got worse in all six
  runs. The ML-Agents documentation covers this case: linear decay is recommended, "however, for
  some cases (e.g. training for an unknown amount of time) this feature can be disabled".
  `epsilon_schedule` is not an independent choice, it defaults to whatever
  `learning_rate_schedule` is, so it was set explicitly rather than left to inherit.
- **Extrinsic `gamma`, 0.99 to 0.998.** `DecisionPeriod: 1` at a 0.02 s timestep is 50 decisions
  per second, and the mean episode is 458 steps, or 9.2 s. At gamma 0.99 the horizon
  `1 / (1 - gamma)` is 100 steps, 2 s. The terminal reward is discounted by `0.99^133 = 0.26`
  across the response alone and `0.99^458 = 0.01` seen from the ITI, so where an agent waits was
  close to invisible to the value function. The documented range of 0.8 to 0.995 is calibrated for
  the ML-Agents example environments, every one of which uses `DecisionPeriod` 5 or 6, where 0.99
  buys a 10 s horizon. Rate-matched to 50 Hz that is `0.99^(1/5) = 0.998`, a 10 s horizon again,
  which covers the episode. So 0.998 is not an unusual value, it is 0.99 restated at this control
  rate.
- **`lambd`, 0.95 to 0.99.** GAE weights n-step returns by `(gamma * lambd)^n`, so its reach is
  `1 / (1 - gamma * lambd)` steps. At gamma 0.998 with lambd 0.95 that is 19 steps, 0.39 s, so
  raising gamma on its own would have changed almost nothing. lambd 0.99 gives 83 steps, 1.7 s.
  The GAE paper's own best values ran 0.92 to 0.99, so this sits at the top of the reported range
  rather than outside it.
- **`time_horizon`, 128 to 512.** At 50 Hz, 128 steps is 2.56 s inside a 9.2 s episode, so almost
  every trajectory was cut early and bootstrapped from the critic instead of running to the
  reward. 512 steps is 10.2 s and covers a whole episode. The 260723 logs fit that reading: value
  loss was flat and very small at about 0.0015 while the critic predicted about +0.05 against
  realised returns near -0.06.
- **`max_steps`, 4e6 to 7e6.** With a constant learning rate nothing ends training on its own, so
  the budget became an explicit choice. Run 0001 used 1e7 and was stopped by hand at 7.12M; from
  run 0002 the config was set to 7e6, about 6.5 hours per model at the observed 1.08M steps per
  hour. ELO was still rising at 7M in some runs, so this is a wall-clock compromise, not a
  converged budget.
- **`even_checkpoints` and `checkpoint_interval`.** Bookkeeping only. `even_checkpoints` makes
  ML-Agents set `checkpoint_interval = max_steps / keep_checkpoints`, giving 40 checkpoints spread
  evenly so one near the ELO plateau can be chosen afterwards. At the old fixed 100000 with
  `keep_checkpoints: 40`, a 7e6 run would have deleted every checkpoint before 3e6.

### What this does not control

Gamma is the change expected to matter most, but it is not isolated. `lambd` and `time_horizon`
both govern how well a decision 325 steps before the reward gets credit for it, so they could
account for some of the shift on their own.

Two further differences are easy to miss.

- `beta_schedule` is `linear` in both, but it decays to zero at `max_steps`, so the same setting
  gives a different trajectory. In 260723 beta was about 6e-5 by 3M. In 260914 it is 0.0021 at 4M
  and only reaches zero at 7M, so entropy pressure is higher for longer, which works against
  commitment rather than for it.
- The build changed with the batch. Nothing in the 260914 build was meant to alter behaviour, but
  it was not verified as identical, so it stays an uncontrolled difference.

Run 0001 of 260914 is also not comparable with the rest of its own batch. It ran under
`max_steps: 1e7`, so its beta schedule was stretched over 10M and ended at 0.0015 rather than
zero, and it was interrupted twice by out-of-memory kills, each of which reset its self-play ELO
to 1200. Its final entropy is 2.43 against 2.14 to 2.33 for runs 0002 to 0004.

## 4. Methods: a stylised model of the game

To ask which settings favour peripheral play, the social game was reduced to a positioning
problem and solved numerically rather than trained.

- Each agent returns during the ITI to a home radius, from 0 (centre) to 16.5 (at a wall),
  aligned with its nearest wall and avoiding the opponent's wall.
- Each trial draws a high wall and a separation from the real distribution.
- Arrival time for each agent and wall is the turning time needed to bring the lit walls into a
  110 deg view, plus the travel time at 10 u/s.
- Both agents then choose high or low at the Nash equilibrium of the resulting 2x2 race game. The
  first to arrive takes its wall's reward, and the other gets nothing.
- Payoffs are scored two ways: raw score, and the discounted return an RL agent maximises, with
  gamma and the step costs included. Returns are measured at slice onset, since discounting over
  the ITI scales every policy's return by the same factor.

Costs can be applied as a time cost on every step (the current build), as a movement cost with
resting free, or as a turning cost.

Validation: with equal rewards and only 180 deg trials, the model gives a wall-sitter a 74.7% win
rate against a centre agent, against 75% worked out by hand.

Limitations: agents have full information about each other, hold a fixed home radius, and never
collide. Learning, memory and exploration are absent. The model indicates direction, not
magnitude. It also overstates gamma, as noted in section 5.

The scripts are `octagon_model.py`, `sweep.py` and `turn.py`, written in a session scratchpad and
not yet in the repo.

## 5. Results

### The geometry favours the wall more than it looks

The distance from one wall to its neighbour is `2R sin(22.5 deg) = 0.77R`, shorter than the
radius. An agent at a wall therefore reaches three walls sooner than a centre agent reaches any.
Because the trial ends at the first trigger entry, reaching any lit wall first also denies the
opponent the other one. Covering walls is worth more than being equidistant from all of them, and
the closer the low reward is to the high, the more that holds.

### Where agents settle, by score

Each row changes one thing and leaves the rest at current values. The last column is the share of
score a wall-sitter takes against a centre agent, where 0.5 is an even split.

| Change | Equilibrium radius | Centrality | Wall-sitter vs centre |
|---|---|---|---|
| Current game (low 0.4, separations 50/25/25, 360 deg/s) | 9 to 14 | 0.54 to 0.29 | 0.47 |
| Low reward 1.0 | 14 | 0.29 | 0.60 |
| Low reward 0.7 | 12 to 14 | 0.39 to 0.29 | 0.54 |
| Low reward 0.1 | 0 | 1.00 | 0.41 |
| All 180 deg trials | 14 | 0.29 | 0.53 |
| Separations 25/25/50 | mostly 14 | 0.29 | 0.50 |
| All 90 deg trials | 9 | 0.54 | 0.47 |
| All 45 deg trials | 3 to 6 | 0.85 to 0.70 | 0.45 |
| Turn speed 90 deg/s | 16.5 | 0.16 | 0.61 |
| Turn speed 45 deg/s | 12 to 14 | 0.39 to 0.29 | 0.69 |

The centre is not the score-optimal position under current settings. The equilibrium sits at
centrality 0.54 to 0.29, straddling the 0.5 classification cut, which means central and peripheral
play are close in value and modest changes can tip the balance.

### What the RL objective adds

| Condition | Equilibrium radius | Centrality |
|---|---|---|
| gamma 0.998, time cost 0.001 (260914) | 9 to 14 | 0.54 to 0.29 |
| gamma 0.99, time cost 0.001 (260723) | 16.5 | 0.16 |
| gamma 0.99, movement cost 0.001, resting free | 16.5 | 0.16 |
| gamma 0.998, movement cost 0.001, resting free | 12 to 15.5 | 0.39 to 0.21 |
| gamma 0.998, movement cost 0.003, resting free | 15.5 to 16.5 | 0.21 to 0.16 |
| gamma 0.998, turning cost 0.01 | 12 to 14 | 0.39 to 0.29 |

Three mechanisms drive these.

- **Discounting.** An agent at a wall is sometimes right on top of a lit trigger and sometimes far
  from one. Discounting rewards that variance, because `gamma^t` falls off fastest early. At
  gamma 0.99 the effective horizon is about 100 steps, close to the 84 steps a centre agent needs
  to reach a wall, so the effect is strong. At 0.998 it mostly disappears.
- **When the cost is paid.** Returning to the centre happens at the start of the episode, about
  325 steps before the reward. Relative to that reward a movement cost counts about twice over at
  gamma 0.998, and about 26 times over at gamma 0.99. A time cost charged on every step has no
  effect on position at all, because resting costs the same as moving.
- **Turning and sight.** Facing inward from a wall, a 110 deg view covers 5 of the other 7 walls.
  From the centre it covers 3 of 8. At 360 deg/s a full look round takes 0.5 s and this barely
  matters. At 90 deg/s it takes 2 s and it decides the game.

### Reality check

The model puts gamma 0.99 agents at the wall, but 260723 produced central and intermediate models.
The direction is right and the magnitude is not, so treat the ordering of these levers as
informative and the predicted positions as optimistic.

## 6. What to try, in order

1. **Restore a movement cost with resting free, and raise it.** Replace the `throw` at
   `OctagonAgent.cs:533` with a no-op so resting simply skips the penalty, rather than deleting
   `&& false` and throwing on every resting step. Then set `step_penalty` to about 0.003 in
   `environment_parameters`. This leaves the task rules identical to the human game and only
   changes what effort costs the agent. Needs a rebuild.
2. **Slow the agent's turning.** `turnSpeed` in `OctagonAgent.cs:35`, at 90 to 180 deg/s. This is
   the strongest lever in the model and changes the agent's body rather than the task. Human turn
   rates should be measured from the logs first to justify a value. Needs a rebuild.
3. **Go back to gamma 0.99, or try 0.995.** Cheapest to run, since it is config only. Based on
   260723, expect a mix of central and intermediate rather than true wall-sitting, so this is
   better combined with 1 or 2 than used alone.
4. **Curriculum on the game parameters.** Train with the low reward at 0.7 to 1.0 and more 180 deg
   trials, then anneal back to the real values. A wall-sitter loses little under the real settings
   (0.47 against 0.50), so the strategy may survive the anneal, but that is untested. Wall
   separation weights can already be driven from the config through `sep_weight_{1,2,4}`, which
   `SelectNewWalls` reads from `EnvironmentParameters`.
5. **Shaping bonus for waiting near a wall during the ITI.** The most direct and the least
   principled. Worth keeping as a fallback, and best annealed to zero once the strategy is
   established.

Varying one of levers 1 to 3 in steps should give a graded set of models from central to
peripheral, which is more useful for modelling player behaviour than a single peripheral batch.

## 7. Measuring the outcome

Run the batch through inference with the fixed trial sequence, then use
`collect_model_data`, `onset_centrality_by_model` and `split_strategies` in
[octagon_analysis/analysis/model_characterisation.py](/home/tom/repos/octagon_analysis/analysis/model_characterisation.py).
The headline number is mean slice-onset centrality per model, with the central and peripheral split
at 0.5 and a Gaussian mixture as a check. Comparing the distribution against 260723 and 260914
shows whether a lever moved the population or only widened it.

## 8. Open questions

- Whether the 260914 shift came from gamma, from `lambd` and `time_horizon`, or from both. One
  run per change would separate them.
- What turn rates human players actually use, which would set a defensible `turnSpeed`.
- Whether a peripheral strategy trained under a curriculum survives annealing back to the real
  game parameters.
- Why the model overstates the gamma effect, given that 260723 did not converge on wall-sitting.
