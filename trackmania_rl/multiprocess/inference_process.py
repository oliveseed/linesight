"""
This file implements a single multithreaded worker that handles a BeamNG game
instance and runs inference-only rollouts. No transitions are collected and no
learner/network updates are performed.
"""

import importlib
import time
from itertools import count
from pathlib import Path

import numpy as np
import torch

from config_files import config_copy
from trackmania_rl.agents import qsac
from trackmania_rl.bng_trajectory import get_vcp_centers_from_file
from trackmania_rl.utilities import set_random_seed


def inference_process_fn(
    game_spawning_lock,
    base_dir: Path,
    save_dir: Path,
    tmi_port: int,
    process_number: int,
):
    from trackmania_rl.tmi_interaction import inference_game_instance_manager as game_instance_manager

    set_random_seed(process_number)

    tmi = game_instance_manager.GameInstanceManager(
        game_spawning_lock=game_spawning_lock,
        max_overall_duration_ms=config_copy.cutoff_rollout_if_race_not_finished_within_duration_ms,
        max_minirace_duration_ms=config_copy.cutoff_rollout_if_no_vcp_passed_within_duration_ms,
        tmi_port=tmi_port,
    )

    inference_network, _ = qsac.make_policy(config_copy.use_jit, is_inference=True)
    try:
        inference_network.load_state_dict(torch.load(f=save_dir / "policy.torch", weights_only=False))
    except Exception as e:
        print("Worker could not load weights, exception:", e)

    inferer = qsac.Inferer(inference_network)
    inferer.is_explo = False

    zone_centers = get_vcp_centers_from_file(trajectory_path=base_dir / "maps" / "nord1.npy", config=config_copy)

    # ========================================================
    # Warmup pytorch
    # ========================================================
    for _ in range(5):
        inferer.infer_actions(
            np.random.randint(low=0, high=255, size=(1, config_copy.H_downsized, config_copy.W_downsized), dtype=np.uint8),
            np.random.rand(config_copy.float_input_dim).astype(np.float32),
            deterministic=True,
        )

    for loop_number in count(1):
        importlib.reload(config_copy)

        tmi.max_minirace_duration_ms = config_copy.cutoff_rollout_if_no_vcp_passed_within_duration_ms

        inference_network.eval()

        rollout_start_time = time.perf_counter()
        tmi.rollout(
            exploration_policy=inferer.get_exploration_action,
            zone_centers=zone_centers,
        )
        rollout_end_time = time.perf_counter()
        rollout_duration = rollout_end_time - rollout_start_time
        print(
            f"[inference] vehicle {tmi_port} loop {loop_number}: "
            f"duration={rollout_duration:.2f}s",
            flush=True,
        )
