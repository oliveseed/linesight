# Linesight — Agent Guide

BeamNG reinforcement learning project using QRSAC — continuous-action SAC with twin **quantile** critics (dropping into IQN infra, replacing the older discrete IQN agent).

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

run_name = "qrsac_test_1"  # change to the run you want to inspect
for run_dir in sorted(glob.glob(f"tensorboard/{run_name}*")):
    ea = EventAccumulator(run_dir)
    ea.Reload()
    print("#####", run_dir, "scalars:", len(ea.Tags()["scalars"]))
    for key in ["loss", "learning_rate", "weight_decay", "sac_alpha", "mean_q", "grad_norm_policy", "grad_norm_critic"]:
        events = ea.Scalars(key)
        print(key, [(events[i].step, round(events[i].value, 6)) for i in (0, len(events) // 2, len(events) - 1)])
```

- `ea.Tags()["scalars"]` lists every tag; `ea.Scalars("<tag>")` returns `(step, value)` events.
- Per-parameter `_L2` tags (`layer_*_L2`, `exp_avg_*_L2`, `exp_avg_sq_*_L2`, `lr_ratio_*_L2`) are RMS values, written every ~5 minutes.
- QRSAC-specific tags written by the learner: `loss`, `critic_loss_mean`, `policy_loss_mean`, `alpha_loss_mean`, `sac_alpha`, `mean_q_*` and `mean_q_starting_frame_*` (aggregated by horizon bucket), `mean_log_pi_*`, `grad_norm_policy`, `grad_norm_critic`, `action_{i}_starting_frame_*` (per-action stats by horizon bucket), plus the generic `avg_Q_*`, `eval_race_finished_*`, `single_zone_reached_*`, `cumul_number_batches_done`, `cumul_number_frames_played`.
- The GUI can be started with `tensorboard --logdir tensorboard` (use `--logdir_spec` to overlay the split directories of one run).

## Architecture

```
scripts/train.py
  ├── collector_process.py   (N processes, one per game instance)
  │     └── game_instance_manager.py  →  bng_interface.py
  └── learner_process.py     (main process, runs training loop)
        └── qsac.py          (Trainer + Inferer)
              SACPolicy, QuantileCritic   (networks)
```

- **Agent (`agents/qsac.py`)** — continuous-action SAC: tanh-Gaussian policy outputs `[steer ∈ [-1,1], throttle ∈ [0,1], brake ∈ [0,1]]` (steer = tanh(u₀), throttle/brake = (tanh(u)+1)/2), twin quantile regression critics (`n_quantiles`, Huber-quantile loss, `_sample_tau` randomizes the tau distribution each iteration), automatic entropy tuning (`log_alpha` scalar param, `target_entropy`), 3 optimizers (policy, critic, alpha) driven by a single `GradScaler` + AMP fp16.
- **Result interface** — `q_values` semantics were replaced: the learner logs `mean_q` (min of mean-critic-values at policy actions) with no quantile-BOUNDS exploration; collectors use the Inferer for probe/greedy actions (`infer_actions(..., deterministic)`).
- **Collectors** run rollouts in game instances, send transitions via `mp.Queue`
- **Learner** trains on collected transitions, updates shared network via `mp.Lock`
- Shared network weights are exchanged via `share_memory()` + `mp.Value` for step count
- The main process IS the learner (saves one CUDA context)

## Config System

`config_files/config.py` → copied to `config_files/config_copy.py` at startup. The learner and collectors `importlib.reload(config_copy)` periodically, so changes to `config_copy.py` mid-training take effect live without restarting. `config.py` changes do NOT affect a running session.

User-specific paths live in `config_files/user_config.py` (TMInterface paths, TMLoader, ports). Edit this once during setup.

## Checkpoints

Saved inside `save/<run_name>/` (see `utilities.save_checkpoint`):

| File | Contents |
|------|----------|
| `policy.torch` / `critic1.torch` / `critic2.torch` | online networks |
| `target_policy.torch` / `target_critic1.torch` / `target_critic2.torch` | target networks (soft-updated) |
| `policy_optimizer.torch` / `critic_optimizer.torch` | optimizer states |
| `alpha.torch` | `{"log_alpha", "alpha_optimizer"}` |
| `scaler.torch` | GradScaler state |

Old IQN checkpoints (`weights1.torch`, `weights2.torch`, `optimizer1.torch`) are **incompatible** with the current agent.

## Key Gotchas

- **Order of optimizer steps vs. backprop (`Trainer.train_on_batch`) is load-bearing** — with a single GradScaler driving policy/critic/alpha optimizers in one autocast region, the critic must be backwarded and stepped **before** policy/alpha losses are backpropagated. `policy_loss` flows through critic1/critic2, so if the critic step were applied after the policy backward, policy gradients would leak into critic parameters on the same batch — a positive feedback loop that makes Q diverge and eventually NaN (this exact bug was found and fixed; see commit `d016080`). Also, the policy/alpha forward must come **after** the critic step (in-place parameter updates would invalidate version counters of graphs built earlier).
- **Autocast cast-cache poisons later grad passes** — do not let the first autocast cast of an online network's parameters happen inside a `no_grad()` region in the same autocast block; the cached half-precision cast severs the graph of subsequent backward passes. Target networks (always `no_grad`) are safe to use first; addressing online online networks only in grad-enabled forwards avoids this.
- **No torch.compile on Windows** — Windows uses `torch.jit.script` instead. Consequences (keep these in mind when editing network code):
  - `ModuleDict` subscripting breaks scripted modules — the network class holds explicit named submodules (`img_head`, `float_feature_extractor`) instead.
  - In-place activations break scripted backward ("a view of a leaf variable that requires grad") — activations are `inplace=False`.
- **CUDA required** — Network runs on CUDA with channels_last memory format; AMP fp16 via `torch.amp.autocast(dtype=Trainer.amp_dtype)`.
- **Multiple game instances** — `gpu_collectors_count` in config defaults to 6. Each needs a separate vehicle on `base_tmi_port + i` in a single BeamNG instance.
- **Signal handling** — `train.py` kills the game process(es) on SIGINT (`taskkill` on Windows, `pkill` on Linux).
- **BeamNG migration in progress** — `game_instance_manager.py` has been rewritten for BeamNG; `bng_interface.py` uses `beamngpy`. The old TMInterface path (`tminterface2.py`) is still present. Current state: The BeamNG implementation is functional but has remnants of the Trackmania implementation. Maintaining compatibility with Trackmania is not required.
- **SET_ACTION wire protocol — continuous actions** — `bng_interface.send_action(steer, throttle, brake)` sends `struct.pack("<I3f", Message.SET_ACTION_CONT, ...)`. The Lua-side handler must decode 3 floats (`<I3f`), not the legacy 7 booleans.
- **Maps are .npy files** in `maps/`, containing centerline coordinates. Loaded via `map_loader.py`. Map cycle is defined in `config_files/config.py`. Map cycle is not used for BeamNG; instead, a single map is hard-coded to load on each experiment.
- **Virtual checkpoints (VCPs)** — The agent uses virtual checkpoints along the centerline for progress tracking. New trajectory handling for BeamNG is in `bng_trajectory.py`.
- **Replay buffer** — Uses `torchrl.data.ReplayBuffer` with optional `PrioritizedSampler`. Buffer resizing and n-step returns are handled in `buffer_utilities.py` and `buffer_management.py`. `buffer_utilities.py` implements buffer_collate_function(), used to customize torchrl's ReplayBuffer.sample() method. The most important customization is our implementation of mini-races, a trick to define Q values as the expected sum of undiscounted rewards in the next 7 seconds. Batch collate returns 7 tensors: `(state_img, state_float, action, rewards, next_state_img, next_state_float, gammas_terminal)`, where `action` is `(batch_size, 3)` continuous.
- **State normalization** — `config_files/state_normalization.py` defines per-feature mean/std arrays. The network normalizes internally in `forward()`.
- **Continuous action space** — 3 actions (`steer, throttle, brake`) from `SACPolicy`; the 11 discrete action list in `config_files/inputs_list.py` is only used by legacy code paths.

## Known Training Issues (TODO)

- **SOLVED — Q divergence / NaN (policy gradients leaking into critics)** — the `qrsac_test_1` run went NaN around step 29,114 with `mean_q` growing unboundedly (≈350 at race level vs. true Q scale ≈±16). Root cause: policy_loss backward happened before the critic optimizer step, depositing "raise Q at policy actions" gradients into the critic parameters on the same batch. Fixed in `Trainer.train_on_batch` (critic-backward → critic-step → freeze critics → policy/alpha backward → unfreeze → policy/alpha steps; verified offline: previously NaN within ~500 steps at lr 1e-3, now 4000+ steps stable). `clip_grad_norm` was also reduced 30 → 1 since pre-clip norms no longer justify 30.
- **Weight decay applied to biases is now skipped** — `custom_weight_decay` (`utilities.py`) decays only parameters with `ndim > 1` and runs inside `torch.no_grad()`. Note this is a deliberate semantics change vs. old IQN runs (which decayed biases too).
- **`adam_epsilon` live-reload writes the wrong key** — legacy issue carried over: the reload path writes `param_group["epsilon"]` instead of `param_group["eps"]`, so mid-run changes have no effect. Unresolved in the QRSAC learner if RAdam is used with `eps`-keyed groups.
- **Schedule concerns** — `lr_schedule` (`config.py:110`) anneals by step; check `config.py` for values relevant to the current run before assuming defaults.

## File Conventions

- `save/` — Checkpoints per run (see Checkpoints section), `accumulated_stats.joblib`
- `tensorboard/` — TensorBoard logs, organized by `run_name`
- `maps/` — Map centerline `.npy` files
- `scripts/tools/` — Utility scripts (gbx-to-vcp conversion, video tools)
- `docs/` — Sphinx documentation source
