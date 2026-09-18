# Linesight — Agent Guide

Trackmania/BeamNG reinforcement learning project using IQN (Implicit Quantile Network).

## Quick Start

```bash
# Install with pixi (preferred)
pixi install

# Or with pip (requires conda for numpy/numba/torch first)
pip install -e .
```

Python 3.10 or 3.11 required (constrained in `pixi.toml` and `setup.py`).

## Linting

```bash
ruff check trackmania_rl/ scripts/ config_files/
ruff format trackmania_rl/ scripts/ config_files/
```

Line length is 140. Ruff config lives in `pyproject.toml`. There is no CI, no typecheck, and no test suite.

## Running

| What | Command | Notes |
|------|---------|-------|
| Training | `python scripts/train.py` | Requires running Trackmania+TMInterface or BeamNG instance |
| Inference | `python scripts/infer.py` | Spawns one collector, no learner |
| Fake vehicle (BeamNG) | `python bng_test_fake_vehicle.py -p 5000` | Test harness, no game needed |
| Record path | `python record_path.py -p 5000` | Records vehicle trajectory to .npy |
| Record author run | `python record_author_run.py -p 5000` | Drives to finish using VCPs |
| State debug view | `python scripts/debug_state_viz.py` | Streams live frame+floats to http://127.0.0.1:8080, no saves |

Training and inference require a running game instance (Trackmania via TMInterface, or BeamNG via bng_interface).

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

## Key Gotchas

- **No torch.compile on Windows** — `iqn.py:155` disables `torch.compile` on non-Linux; Windows uses `torch.jit.script` instead.
- **CUDA required** — Network runs on CUDA with channels_last memory format.
- **Multiple game instances** — `gpu_collectors_count` in config defaults to 6. Each needs a separate vehicle on `base_tmi_port + i` in a single BeamNG instance.
- **Signal handling** — `train.py` kills all `TmForever.exe` on SIGINT (`taskkill` on Windows, `pkill` on Linux).
- **BeamNG migration in progress** — `game_instance_manager.py` has been rewritten for BeamNG; `bng_interface.py` uses `beamngpy`. The old TMInterface path (`tminterface2.py`) is still present. Current state: The BeamNG implementation is functional but has remnants of the Trackmania implementation. Maintaining compatibility with Trackmania is not required.
- **Maps are .npy files** in `maps/`, containing centerline coordinates. Loaded via `map_loader.py`. Map cycle is defined in `config_files/config.py`. Map cycle is not used for BeamNG; instead, a single map is hard-coded to load on each experiment.
- **Virtual checkpoints (VCPs)** — The agent uses virtual checkpoints along the centerline for progress tracking. New trajectory handling for BeamNG is in `bng_trajectory.py`.
- **Replay buffer** — Uses `torchrl.data.ReplayBuffer` with optional `PrioritizedSampler`. Buffer resizing and n-step returns are handled in `buffer_utilities.py` and `buffer_management.py`.
- **State normalization** — `config_files/state_normalization.py` defines per-feature mean/std arrays. The network normalizes internally in `forward()`.
- **11 discrete actions** — Defined in `config_files/inputs_list.py` (throttle/brake/steer combinations).

## File Conventions

- `save/` — Checkpoints, weights (`weights1.torch`, `weights2.torch`), `accumulated_stats.joblib`, best runs
- `tensorboard/` — TensorBoard logs, organized by `run_name`
- `maps/` — Map centerline `.npy` files
- `scripts/tools/` — Utility scripts (gbx-to-vcp conversion, video tools)
- `docs/` — Sphinx documentation source
