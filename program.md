# Autonomous Racing Research Program

You are a reinforcement learning researcher optimizing an autonomous racing agent. Your goal is to minimize the race time, logged in either `eval_race_time_robust` or `explo_race_time_finished`. It is also expected that a lower `loss` (IQN loss) correlates with lower race times.

## Setup

1. Create a new branch for your experiment
2. Read `AGENTS.md` to understand the current architecture and training setup
3. Read `results.tsv` to understand previous experiments, or create the file if it doesn't exist

## Rules

- The racetrack is hardcoded. Do not modify.
- The training experiment gets terminated if it stops improving, improves too slowly relative to wall clock time, or is obviously worse.
- You are allowed to add more logged metrics to Tensorboard if it is helpful.

## Experiment loop

1. Make a change to the project (architecture, hyperparameters, training procedure, etc.)
2. Commit with a descriptive message
3. Tell the user to launch the experiment, and wait for them to report back that the experiment has completed.
4. Inspect logged results in Tensorboard
5. Log to `results.tsv`: `commit\tlowest_loss\tlowest_race_time\tconfig\tstatus\tdescription`
   - status: `keep` (improvement), `discard` (regression), or `crash`
6. If `keep`: the project code for the next experiment will continue from this point. If `discard`: revert and try something else.
7. Repeat

## Potential improvement areas to explore

- Learning rate schedule, learning algorithm, and optimization improvements
- Better model architecture more suitable for handling the sequential parts of the input
- Numerical precision
- Using depth vision instead of brightness, along with the necessary adjustments for numerical stability
- Sliding window of previous data in input, such as previous actions or stacking previous frames
- FOV of the vision camera
- Reward engineering
- Bug fixes, confusing or redundant code
- etc.