# Training hyperparameters: schedules, step budget and what to vary

Conclusions from comparing 260723 and 260914, written 2026-09-28. Evidence is in
`training_convergence_260723_vs_260914.ipynb` (training curves) and
`octagon_analysis/demos/strategy_characterisation/models/competitive/model_characterisation_demo260914.ipynb`
(behaviour).

## Schedules

**Use constant `learning_rate_schedule` and `epsilon_schedule`, and keep `beta_schedule` linear.**

A linear schedule decays its value to zero at `max_steps`, by
`value = initial * (1 - step / max_steps)`, recomputed from whatever `max_steps` is set when a run
resumes. Constant holds the value throughout.

Constant learning rate suits exploratory work for two reasons. A run can be extended without
committing to a step budget up front, which matters because a high-gamma recipe takes far longer to
converge than expected. More importantly, **every run sits in the same training regime at any given
step**, so batches can be compared at matched steps without the schedule position itself becoming
the difference.

That second point is not hypothetical. 260723 used linear schedules with `max_steps` 4e6 and 260914
used constant with 7e6, so at a "matched" 4e6 the 260723 runs were annealed to a stop (learning rate
1.8e-5) while the 260914 runs were mid-flight at 1.5e-4. Comparing them there compares training
regimes, not hyperparameters.

Keep `beta_schedule` linear. The entropy bonus decaying to zero is what moves a policy from
exploring to committing, and it is doing real work: entropy fell from 3.03 to 2.24 across 260914,
which is what saturates High-wall choice. Setting beta constant risks policies that never commit.

**The catch: `beta_schedule: linear` keys off `max_steps`, so `max_steps` must be identical across
every arm of a comparison.** Otherwise the entropy schedule becomes exactly the confound that the
constant learning rate was chosen to avoid. If a batch is extended, extend every arm together.

Annealing the learning rate was expected to give a more settled final policy, but no such effect is
measurable here. Detrended wobble in episode length over each run's final 1e6 steps is 12.6 for the
annealed 260723 runs and 12.8 for the constant 260914 runs, despite an eight-fold difference in
learning rate. Episode length is coarse, so this is absence of evidence rather than evidence of
absence, but there is no reason to pay for annealing during exploration.

**For polished models later**, resume the chosen run with `learning_rate_schedule: linear` and
`max_steps` about 1.15x its current step. That anneals from roughly 13% of the initial learning rate
down to zero without committing to a budget in advance. The mechanism is confirmed by 260723's own
extension, where raising `max_steps` from 3e6 to 4e6 restarted the decay exactly as the formula
predicts. Whether it improves anything is untested; one resumed run against its own unannealed
checkpoint would settle it cheaply.

## Step budget

**Use 7e6 for all arms.** One run takes hours at that length, so this is the main compute decision.

260723 converged by about 3e6: its episode length levelled off while the learning rate was still
near half peak, and its ELO is flat over its last million steps. It cannot be improved by spending
more steps within its own recipe.

260914 had not converged at 4e6 and had still not converged at 7e6. ELO is still rising and episode
length still falling at the end of every run. Raising gamma slows convergence substantially, which
is the single most important practical consequence of the change.

So 7e6 is a floor for a high-gamma recipe rather than a comfortable budget, and 4e6 would anneal or
truncate a half-learned policy. A low-gamma arm reaches its plateau long before 7e6 and simply sits
there; with a constant learning rate the extra steps cost compute but distort nothing, and matching
`max_steps` keeps the beta schedule identical across arms. That trade is worth making.

Behaviour keeps improving well past the point where choice has saturated. Between 4e6 and 7e6 the
260914 models changed `p_high` barely at all but improved route directness in all eight models
measured (`path_ratio` 1.278 to 1.168). **Efficiency is learned long after choice is.** A batch
stopped early looks like it learned the wrong thing when it has only learned half of it.

## Seeds

`sequential_runs` defaults to `DEFAULT_TRAINING_SEEDS` in `trainer_and_simulator_functions.py`, the
50 seeds starting with the 30 260914 was trained on, indexed by absolute model number. Model 0007 of any batch therefore
trains on the same seed as `260914_0007`, which makes a comparison between batches paired rather
than unpaired and removes a large source of variance for free.

Leave `seeds` unset. Pass `seeds=None` only for a batch that is deliberately not meant to be
compared with others; it will draw fresh random seeds and print a warning. Running past model 0050
needs the list extended.

## What to vary, and what to hold

Vary one thing per batch. Hold everything else at the 260914 values, including the build, the trial
sequence, `n_envs`, the schedules, `max_steps` and the seeds.

**gamma, lambd and `time_horizon` are not independent knobs.** They were changed together as a
matched set: 1/(1-0.99) = 100 which pairs with `time_horizon` 128, and 1/(1-0.998) = 500 which pairs
with 512; the GAE horizon 1/(1-lambd*gamma) moved from 16.8 to 83.4 alongside. A full grid over the
three mostly produces incoherent cells, since gamma 0.998 with `time_horizon` 128 truncates the
horizon the gamma is paying for. Treat them as one credit-assignment package and break the pairing
deliberately, one piece at a time.

The current arm is **260929** (`SocialConfig260929.yaml`): gamma 0.998 with `lambd` and
`time_horizon` at their 260723 values. If it reproduces the 260914 outcome, gamma alone is
sufficient. If not, add `lambd` back, then `time_horizon`.

## What counts as a better batch

Judge a batch on the fraction of seeds that produce a usable model, not only on the mean of the ones
that worked. That is where gamma had its largest effect: 24 of 55 260723 runs (44%) landed in the
peripheral standing-still strategy, against 1 of 30 (3%) for 260914. Whole-batch score rate rose
from 0.395 to 0.458 and its spread across models nearly halved.

For the models themselves, `score_rate` is the best single self-play measure because it is the only
one that accounts for time. Note that `mean_score` is not a measure of skill: in self-play each
agent wins about half the trials, so mean score follows from `0.5 * (0.4 + 0.6 * p_high)` and every
model in both batches lands within 0.05 of that. Self-play also under-penalises slowness, since two
equally slow agents lose nothing to each other. A tournament against different opponents is the only
conclusive comparison.
