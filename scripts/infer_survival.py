# =======================================================================================================================
# Handle configuration file before running the actual content of infer_survival.py
# =======================================================================================================================
"""
Inference does NOT copy config.py over config_copy.py by default (unlike train.py/infer.py):
config_copy.py may hold live tweaks for an ongoing run (it is reloaded periodically by running
processes), and overwriting it would silently destroy those tweaks. Pass --copy-config to get
the usual infer.py behavior of resetting config_copy.py from config.py first.
"""

import shutil
from pathlib import Path


def copy_configuration_file():
    base_dir = Path(__file__).resolve().parents[1]
    shutil.copyfile(
        base_dir / "config_files" / "config.py",
        base_dir / "config_files" / "config_copy.py",
    )

# =======================================================================================================================
# Actual start of infer_survival.py
# =======================================================================================================================

import argparse
import importlib
import json
import signal
import sys
import time
from datetime import datetime
from itertools import count

import torch
import torch.multiprocessing as mp
from art import tprint

from config_files import config_copy
from trackmania_rl.utilities import set_random_seed

# noinspection PyUnresolvedReferences
torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)
torch.set_float32_matmul_precision("high")


def signal_handler(sig, frame):
    print("Received SIGINT signal. Killing all collector processes.")

    for child in mp.active_children():
        child.kill()

    tprint("Bye bye!", font="tarty1")
    sys.exit()


def worker_fn(
    game_spawning_lock,
    base_dir: Path,
    save_dir: Path,
    output_dir: Path,
    tmi_port: int,
    process_number: int,
    n_rollouts: int,
    rollout_counter,
    counter_lock,
    map_path: Path,
    weights_filename: str,
):
    import numpy as np

    from trackmania_rl.agents import iqn
    from trackmania_rl.bng_trajectory import get_vcp_centers_from_file
    from trackmania_rl.tmi_interaction import inference_game_instance_manager as game_instance_manager

    set_random_seed(process_number)

    tmi = game_instance_manager.GameInstanceManager(
        game_spawning_lock=game_spawning_lock,
        max_overall_duration_ms=config_copy.cutoff_rollout_if_race_not_finished_within_duration_ms,
        max_minirace_duration_ms=config_copy.cutoff_rollout_if_no_vcp_passed_within_duration_ms,
        tmi_port=tmi_port,
    )

    inference_network, _ = iqn.make_untrained_iqn_network(config_copy.use_jit, is_inference=True)
    # Fail loudly rather than silently evaluate random weights.
    inference_network.load_state_dict(
        torch.load(f=save_dir / weights_filename, weights_only=False)
    )

    inferer = iqn.Inferer(inference_network, config_copy.iqn_k, config_copy.tau_epsilon_boltzmann)
    inferer.epsilon = 0.0
    inferer.epsilon_boltzmann = 0.0
    inferer.tau_epsilon_boltzmann = config_copy.tau_epsilon_boltzmann
    inferer.is_explo = False

    zone_centers = get_vcp_centers_from_file(trajectory_path=map_path, config=config_copy)

    # ========================================================
    # Warmup pytorch
    # ========================================================
    for _ in range(5):
        inferer.infer_network(
            np.random.randint(low=0, high=255, size=(1, config_copy.H_downsized, config_copy.W_downsized), dtype=np.uint8),
            np.random.rand(config_copy.float_input_dim).astype(np.float32),
        )

    part_path = output_dir / f"rollouts_part_{process_number}.jsonl"

    for _ in count(1):
        importlib.reload(config_copy)

        tmi.max_minirace_duration_ms = config_copy.cutoff_rollout_if_no_vcp_passed_within_duration_ms

        inference_network.eval()

        # Claim a rollout slot up front so that exactly n_rollouts are run across all workers.
        with counter_lock:
            if rollout_counter.value >= n_rollouts:
                break
            rollout_id = rollout_counter.value
            rollout_counter.value += 1

        rollout_start_time = time.perf_counter()
        result = tmi.rollout(
            exploration_policy=inferer.get_exploration_action,
            zone_centers=zone_centers,
        )
        rollout_end_time = time.perf_counter()

        result["rollout_id"] = rollout_id
        result["worker"] = process_number
        result["wall_time_s"] = rollout_end_time - rollout_start_time

        with open(part_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

        print(
            f"[infer_survival] rollout {rollout_id + 1}/{n_rollouts} (vehicle {tmi_port}): "
            f"reason={result['truncation_reason']} furthest_zone={result['furthest_zone_idx']} "
            f"end_zone={result['end_zone_idx']} race_time={result['race_time_ms'] / 1000:.1f}s "
            f"wall_time={result['wall_time_s']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    if "--copy-config" in sys.argv:
        copy_configuration_file()
        importlib.reload(config_copy)
        print("Reset config_copy.py from config.py (--copy-config).")
    else:
        print("Using config_files/config_copy.py as-is. Pass --copy-config to reset it from config.py first.")

    base_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description=(
            "Run n (default 100) greedy inference rollouts from the beginning of the map, "
            "save per-rollout truncation stats to disk for offline plotting "
            "(see scripts/plot_inference_survival.py)."
        ),
    )
    parser.add_argument("--n-rollouts", type=int, default=100, help="Total number of rollouts to run (default: 100)")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=config_copy.gpu_collectors_count,
        help="Number of parallel vehicles/processes to use (default: config gpu_collectors_count)",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=base_dir / "save" / config_copy.run_name,
        help="Directory containing the weights to load (default: save/<run_name>)",
    )
    parser.add_argument("--weights", type=str, default="weights1.torch", help="Weights filename inside --save-dir (default: weights1.torch)")
    parser.add_argument(
        "--map",
        type=Path,
        default=base_dir / "maps" / "nord1.npy",
        help="Centerline .npy used for the rollouts (default: maps/nord1.npy)",
    )
    parser.add_argument(
        "--copy-config",
        action="store_true",
        help="Reset config_files/config_copy.py from config.py before starting (default: keep config_copy.py as-is)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write rollouts.json and part files (default: save/inference_survival/<run_name>/<timestamp>/)",
    )
    args = parser.parse_args()
    args.save_dir = args.save_dir.resolve()
    args.map = args.map.resolve()

    if args.num_workers < 1:
        parser.error("--num-workers must be >= 1")
    # if args.num_workers > 6:
    #     print(
    #         f"WARNING: --num-workers={args.num_workers} exceeds the 6 vehicles BeamNG spawns by default\n"
    #         "(some_vehicle_0..5 on ports base_tmi_port+0..5). Extra workers will crash at startup\n"
    #         "unless you spawned additional vehicles.\n"
    #     )

    random_seed = 444
    set_random_seed(random_seed)

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else base_dir / "save" / "inference_survival" / config_copy.run_name / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = args.save_dir / args.weights
    if not weights_path.is_file():
        print(f"ERROR: weights not found: {weights_path}\n")
        print("Available runs under save/ with weights1.torch:")
        for run_dir in sorted((base_dir / "save").glob("*/weights1.torch")):
            print(f"  {run_dir.parent.name}")
        sys.exit(2)

    tprint("Linesight", font="tarty1")
    print(f"\nInference survival run: {args.n_rollouts} rollouts on {args.num_workers} vehicles\n")
    print(f"Weights:   {args.save_dir / args.weights}")
    print(f"Map:       {args.map}")
    print(f"Output:    {output_dir}\n")

    game_spawning_lock = mp.Lock()
    rollout_counter = mp.Value("i", 0)
    counter_lock = mp.Lock()

    collector_processes = [
        mp.Process(
            target=worker_fn,
            args=(
                game_spawning_lock,
                base_dir,
                args.save_dir,
                output_dir,
                config_copy.base_tmi_port + process_number,
                process_number,
                args.n_rollouts,
                rollout_counter,
                counter_lock,
                args.map,
                args.weights,
            ),
        )
        for process_number in range(args.num_workers)
    ]
    for collector_process in collector_processes:
        collector_process.start()

    for collector_process in collector_processes:
        collector_process.join()

    for i, collector_process in enumerate(collector_processes):
        if collector_process.exitcode != 0:
            print(f"WARNING: worker {i} exited with code {collector_process.exitcode}")

    # ========================================================
    # Merge per-worker part files into a single rollouts.json
    # ========================================================
    rollouts = []
    for part_path in sorted(output_dir.glob("rollouts_part_*.jsonl")):
        with open(part_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rollouts.append(json.loads(line))
    rollouts.sort(key=lambda r: r["rollout_id"])

    payload = {
        "run_name": config_copy.run_name,
        "weights_file": str(args.save_dir / args.weights),
        "map": str(args.map),
        "n_rollouts_requested": args.n_rollouts,
        "n_rollouts": len(rollouts),
        "created": datetime.now().isoformat(timespec="seconds"),
        "rollouts": rollouts,
    }
    data_path = output_dir / "rollouts.json"
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # ========================================================
    # Summary
    # ========================================================
    reasons = {}
    for r in rollouts:
        reasons[r["truncation_reason"]] = reasons.get(r["truncation_reason"], 0) + 1
    finished = [r for r in rollouts if r["race_finished"]]
    race_times = [r["race_time_ms"] / 1000 for r in rollouts]

    print("\n================ Inference survival summary ================")
    print(f"Rollouts completed:  {len(rollouts)}/{args.n_rollouts}")
    print(f"Truncation reasons:  {reasons}")
    print(f"Finishes:            {len(finished)}/{len(rollouts)}")
    if race_times:
        print(f"Race time (all):     mean={sum(race_times) / len(race_times):.1f}s  min={min(race_times):.1f}s  max={max(race_times):.1f}s")
    if finished:
        ft = [r["race_time_ms"] / 1000 for r in finished]
        print(f"Race time (finish):  mean={sum(ft) / len(ft):.1f}s  median={sorted(ft)[len(ft) // 2]:.1f}s")
    print(f"\nData written to:     {data_path}")
    print(f'Plot with:           python scripts/plot_inference_survival.py --data "{data_path}"  (with the project\'s environment)')

    if len(rollouts) < args.n_rollouts:
        print(
            f"\nWARNING: only {len(rollouts)}/{args.n_rollouts} rollouts were completed. "
            "Some workers crashed (check the logs above). The partial data was still written; "
            "fix the game setup and re-run to get the full set."
        )
        sys.exit(3)
