# Linesight — Agent Guide

BeamNG reinforcement learning project using IQN (Implicit Quantile Network).

## Quick Start

```bash
# pip (requires conda for numpy/numba/torch first)
pip install -e .
```

Python 3.10 or 3.11 required (constrained in `pixi.toml` and `setup.py`). When running scripts, use the existing virtual environment which already has all requirements installed.

## Linting

```bash
ruff check trackmania_rl/ scripts/ config_files/
ruff format trackmania_rl/ scripts/ config_files/
```

Line length is 140. Ruff config lives in `pyproject.toml`. There is no CI, no typecheck, and no test suite.

## Running

| What | Command | Notes |
|------|---------|-------|
| Training | `python scripts/train.py` | Requires running BeamNG instance |
| Inference | `python scripts/infer.py` | Spawns one collector, no learner |
| Record path | `python record_path.py -p 5000` | Records vehicle trajectory to .npy |
| State debug view | `python scripts/debug_state_viz.py` | Streams live frame+floats to http://127.0.0.1:8080, no saves |

Training and inference require a running game instance (BeamNG via bng_interface).

## Inspecting TensorBoard Logs

Logs are written to `tensorboard/<run_name><suffix>`. A single logical run is **split across multiple directories** by `tensorboard_suffix_schedule` (`config.py:116`): `""` (0–6M steps), `_2` (6–15M), `_3` (15–30M), `_4` (30–45M), etc. To see a whole run, iterate over every directory matching the run name.

Reading uses `EventAccumulator` from `tensorboard`. Run it with the project's virtual environment (activate it, or call its `Scripts\python.exe` directly) rather than the `python` on `PATH`, which may not have `torch`/`tensorboard` installed.

```python
import glob

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run_name = "test_crap_21"  # change to the run you want to inspect
for run_dir in sorted(glob.glob(f"tensorboard/{run_name}*")):
    ea = EventAccumulator(run_dir)
    ea.Reload()
    print("#####", run_dir, "scalars:", len(ea.Tags()["scalars"]))
    for key in ["loss", "loss_test", "grad_norm_history_median", "grad_norm_history_max", "learning_rate", "weight_decay"]:
        events = ea.Scalars(key)
        print(key, [(events[i].step, round(events[i].value, 6)) for i in (0, len(events) // 2, len(events) - 1)])
```

- `ea.Tags()["scalars"]` lists every tag; `ea.Scalars("<tag>")` returns `(step, value)` events.
- Per-parameter `_L2` tags (`layer_*_L2`, `lr_ratio_*_L2`, `exp_avg_*_L2`, `exp_avg_sq_*_L2`) are RMS values, written every ~5 minutes.
- Commonly useful tags: `loss`, `loss_test`, `grad_norm_history_{median,max,d98}`, `learning_rate`, `weight_decay`, `avg_Q_*`, `eval_race_finished_*`, `single_zone_reached_*`, `cumul_number_batches_done`, `cumul_number_frames_played`.
- The GUI can be started with `tensorboard --logdir tensorboard` (use `--logdir_spec` to overlay the split directories of one run).

## Architecture

```
scripts/train.py
  ├── collector_process.py   (N processes, one per game instance)
  │     └── game_instance_manager.py  →  bng_interface.py / tminterface2.py
  └── learner_process.py     (main process, runs training loop)
        └── iqn.py           (Trainer + Inferer + IQN_Network)
```

- **Collectors** run rollouts in game instances, send transitions via `mp.Queue`
- **Learner** trains on collected transitions, updates shared network via `mp.Lock`
- Shared network weights are exchanged via `share_memory()` + `mp.Value` for step count
- The main process IS the learner (saves one CUDA context)

## Config System

`config_files/config.py` → copied to `config_files/config_copy.py` at startup. The learner and collectors `importlib.reload(config_copy)` periodically, so changes to `config_copy.py` mid-training take effect live without restarting. `config.py` changes do NOT affect a running session.

User-specific paths live in `config_files/user_config.py` (TMInterface paths, TMLoader, ports). Edit this once during setup.

## Known Training Issues (TODO)

Audited from the `test_crap_21` run (TensorBoard) and a code review. These are unresolved unless marked otherwise.

- **Loss is summed, not averaged, over the batch** — `iqn.py:318` applies `torch.sum(...)` to a `batch_size=512` loss, so gradients are ~512× larger than a mean-reduced loss. The logged pre-clip grad norms are ~2k–20k, which means `clip_grad_norm=30` binds on *every* batch (turning the update into a normalized-gradient step) and `clip_grad_value=1000` is a no-op (elements are already ≤ 30). Fix: mean-reduce (or divide by `batch_size`) and re-tune `clip_grad_norm`; drop or raise `clip_grad_value` (`config.py:143`).
- **Weight decay is applied to biases too** — `custom_weight_decay` (`utilities.py:86`) multiplies every `state_dict()` entry, i.e. weights *and* biases. The input-normalization tensors (`float_inputs_mean/std`) are plain attributes, not buffers, so they are *not* decayed. Fix: skip parameters with `ndim <= 1`.
- **Weight decay / LR coupling is confusing (but not a math bug)** — `weight_decay = weight_decay_lr_ratio * learning_rate` then `p *= (1 - weight_decay)` (`learner_process.py:268,529`) is algebraically equal to decoupled AdamW with λ = 1/50. Problems are cosmetic/design: the logged `weight_decay` is the per-step decay (~1e-6), not the coefficient; it vanishes as lr → 0; and `weight_decay_lr_ratio` is really λ. Consider decoupling and renaming.
- **`adam_epsilon=1e-4` is too high and cannot be tuned live** — default is 1e-8 (`config.py:133`). At 1e-4 it is comparable to `sqrt(exp_avg_sq)` for many parameters (see `lr_ratio_*_L2` saturating near 1e4). More importantly, the reload path writes the **wrong key**: `param_group["epsilon"]` instead of `param_group["eps"]` (`learner_process.py:291`). The saved `optimizer1.torch` contains both keys, confirming that mid-run changes to `adam_epsilon` have no effect. Fix the key and lower epsilon.
- **Schedule concerns** — `lr_schedule` (`config.py:110`) anneals to 1e-5 at 15M, before the policy can complete laps consistently.

## Key Gotchas

- **No torch.compile on Windows** — `iqn.py:155` disables `torch.compile` on non-Linux; Windows uses `torch.jit.script` instead.
- **CUDA required** — Network runs on CUDA with channels_last memory format.
- **Multiple game instances** — `gpu_collectors_count` in config defaults to 6. Each needs a separate vehicle on `base_tmi_port + i` in a single BeamNG instance.
- **Signal handling** — `train.py` kills all `TmForever.exe` on SIGINT (`taskkill` on Windows, `pkill` on Linux).
- **BeamNG migration in progress** — `game_instance_manager.py` has been rewritten for BeamNG; `bng_interface.py` uses `beamngpy`. The old TMInterface path (`tminterface2.py`) is still present. Current state: The BeamNG implementation is functional but has remnants of the Trackmania implementation. Maintaining compatibility with Trackmania is not required.
- **Maps are .npy files** in `maps/`, containing centerline coordinates. Loaded via `map_loader.py`. Map cycle is defined in `config_files/config.py`. Map cycle is not used for BeamNG; instead, a single map is hard-coded to load on each experiment.
- **Virtual checkpoints (VCPs)** — The agent uses virtual checkpoints along the centerline for progress tracking. New trajectory handling for BeamNG is in `bng_trajectory.py`.
- **Replay buffer** — Uses `torchrl.data.ReplayBuffer` with optional `PrioritizedSampler`. Buffer resizing and n-step returns are handled in `buffer_utilities.py` and `buffer_management.py`. `buffer_utilities.py` implements buffer_collate_function(), used to customize torchrl’s ReplayBuffer.sample() method. The most important customization is our implementation of mini-races, a trick to define Q values as the expected sum of undiscounted rewards in the next 7 seconds.
- **State normalization** — `config_files/state_normalization.py` defines per-feature mean/std arrays. The network normalizes internally in `forward()`.
- **11 discrete actions** — Defined in `config_files/inputs_list.py` (throttle/brake/steer combinations).

## File Conventions

- `save/` — Checkpoints, weights (`weights1.torch`, `weights2.torch`), `accumulated_stats.joblib`, best runs
- `tensorboard/` — TensorBoard logs, organized by `run_name`
- `maps/` — Map centerline `.npy` files
- `scripts/tools/` — Utility scripts (gbx-to-vcp conversion, video tools)
- `docs/` — Sphinx documentation source
