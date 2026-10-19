"""
This file implements the main training loop, tensorboard statistics tracking, etc...
"""

import copy
import importlib
import math
import random
import sys
import time
import typing
from collections import defaultdict
from datetime import datetime
from multiprocessing.connection import wait
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torchrl.data.replay_buffers import PrioritizedSampler

from config_files import config_copy
from trackmania_rl import buffer_management, utilities
from trackmania_rl.agents import qsac
from trackmania_rl.analysis_metrics import (
    distribution_curves,
    highest_prio_transitions,
    loss_distribution,
    race_time_left_curves,
)
from trackmania_rl.buffer_utilities import make_buffers, resize_buffers
from trackmania_rl.map_reference_times import reference_times


def learner_process_fn(
    rollout_queues,
    uncompiled_shared_network,
    shared_network_lock,
    shared_steps: mp.Value,
    base_dir: Path,
    save_dir: Path,
    tensorboard_base_dir: Path,
):
    layout_version = "lay_mono"
    SummaryWriter(log_dir=str(tensorboard_base_dir / layout_version)).add_custom_scalars(
        {
            layout_version: {
                # "eval_agg_ratio": [
                #     "Multiline",
                #     [
                #         "eval_agg_ratio_trained_author",
                #         "eval_agg_ratio_blind_author",
                #     ],
                # ],
                # "eval_ratio_trained_author": [
                #     "Multiline",
                #     [
                #         "eval_ratio_trained_author",
                #     ],
                # ],
                # "eval_ratio_blind_author": [
                #     "Multiline",
                #     [
                #         "eval_ratio_blind_author",
                #     ],
                # ],
                "eval_race_time_robust": [
                    "Multiline",
                    [
                        "eval_race_time_robust",
                    ],
                ],
                "explo_race_time_finished": [
                    "Multiline",
                    [
                        "explo_race_time_finished",
                    ],
                ],
                "loss": ["Multiline", ["loss$", "loss_test$"]],
                "q_and_alpha": [
                    "Multiline",
                    ["mean_q", "sac_alpha"],
                ],
                "single_zone_reached": [
                    "Multiline",
                    [
                        "single_zone_reached",
                    ],
                ],
                "grad_norm_history": [
                    "Multiline",
                    [
                        "grad_norm_history_d9",
                        "grad_norm_history_d98",
                    ],
                ],
                "priorities": [
                    "Multiline",
                    [
                        "priorities",
                    ],
                ],
            },
        }
    )

    # ========================================================
    # Create new stuff
    # ========================================================

    (
        policy_online,
        uncompiled_policy_online,
        critic1,
        critic2,
        target_policy,
        target_critic1,
        target_critic2,
    ) = qsac.make_untrained_qsac_networks(config_copy.use_jit, is_inference=False)

    print(policy_online)
    utilities.count_parameters(policy_online)

    accumulated_stats: defaultdict[str, typing.Any] = defaultdict(int)
    accumulated_stats["alltime_min_ms"] = {}
    accumulated_stats["rolling_mean_ms"] = {}
    previous_alltime_min = None
    time_last_save = time.perf_counter()
    queue_check_order = list(range(len(rollout_queues)))
    rollout_queue_readers = [q._reader for q in rollout_queues]
    time_waited_for_workers_since_last_tensorboard_write = 0
    time_training_since_last_tensorboard_write = 0
    time_testing_since_last_tensorboard_write = 0

    # ========================================================
    # Load existing stuff
    # ========================================================
    # noinspection PyBroadException
    try:
        uncompiled_policy_online.load_state_dict(torch.load(f=save_dir / "policy.torch", weights_only=False))
        critic1.load_state_dict(torch.load(f=save_dir / "critic1.torch", weights_only=False))
        critic2.load_state_dict(torch.load(f=save_dir / "critic2.torch", weights_only=False))
        target_policy.load_state_dict(torch.load(f=save_dir / "target_policy.torch", weights_only=False))
        target_critic1.load_state_dict(torch.load(f=save_dir / "target_critic1.torch", weights_only=False))
        target_critic2.load_state_dict(torch.load(f=save_dir / "target_critic2.torch", weights_only=False))
        print(" =====================     Learner weights loaded !     ============================")
    except:
        print(" Learner could not load weights")

    with shared_network_lock:
        uncompiled_shared_network.load_state_dict(uncompiled_policy_online.state_dict())

    # noinspection PyBroadException
    try:
        accumulated_stats = joblib.load(save_dir / "accumulated_stats.joblib")
        shared_steps.value = accumulated_stats["cumul_number_frames_played"]
        print(" =====================      Learner stats loaded !      ============================")
    except:
        print(" Learner could not load stats")

    if "rolling_mean_ms" not in accumulated_stats.keys():
        # Temporary to preserve compatibility with old runs that doesn't have this feature. To be removed later.
        accumulated_stats["rolling_mean_ms"] = {}

    accumulated_stats["cumul_number_single_memories_should_have_been_used"] = accumulated_stats["cumul_number_single_memories_used"]
    transitions_learned_last_save = accumulated_stats["cumul_number_single_memories_used"]
    neural_net_reset_counter = 0
    single_reset_flag = config_copy.single_reset_flag

    optimizer_policy = torch.optim.RAdam(
        policy_online.parameters(),
        lr=utilities.from_exponential_schedule(config_copy.lr_schedule, accumulated_stats["cumul_number_frames_played"]) * config_copy.policy_lr_ratio,
        eps=config_copy.adam_epsilon,
        betas=(config_copy.adam_beta1, config_copy.adam_beta2),
    )
    optimizer_critic = torch.optim.RAdam(
        list(critic1.parameters()) + list(critic2.parameters()),
        lr=utilities.from_exponential_schedule(config_copy.lr_schedule, accumulated_stats["cumul_number_frames_played"]) * config_copy.critic_lr_ratio,
        eps=config_copy.adam_epsilon,
        betas=(config_copy.adam_beta1, config_copy.adam_beta2),
    )
    log_alpha = torch.tensor(math.log(config_copy.alpha_initial), device="cuda", requires_grad=True)
    optimizer_alpha = torch.optim.RAdam(
        [log_alpha],
        lr=utilities.from_exponential_schedule(config_copy.lr_schedule, accumulated_stats["cumul_number_frames_played"]) * config_copy.alpha_lr_ratio,
        eps=config_copy.adam_epsilon,
        betas=(config_copy.adam_beta1, config_copy.adam_beta2),
    )

    scaler = torch.amp.GradScaler("cuda")
    memory_size, memory_size_start_learn = utilities.from_staircase_schedule(
        config_copy.memory_size_schedule, accumulated_stats["cumul_number_frames_played"]
    )
    buffer, buffer_test = make_buffers(memory_size)
    offset_cumul_number_single_memories_used = memory_size_start_learn * config_copy.number_times_single_memory_is_used_before_discard

    # noinspection PyBroadException
    try:
        optimizer_policy.load_state_dict(torch.load(f=save_dir / "policy_optimizer.torch", weights_only=False))
        optimizer_critic.load_state_dict(torch.load(f=save_dir / "critic_optimizer.torch", weights_only=False))
        alpha_state = torch.load(f=save_dir / "alpha.torch", weights_only=False)
        with torch.no_grad():
            log_alpha.copy_(alpha_state["log_alpha"])
        optimizer_alpha.load_state_dict(alpha_state["alpha_optimizer"])
        scaler.load_state_dict(torch.load(f=save_dir / "scaler.torch", weights_only=False))
        print(" =========================     Optimizer loaded !     ================================")
    except:
        print(" Could not load optimizer")

    tensorboard_suffix = utilities.from_staircase_schedule(
        config_copy.tensorboard_suffix_schedule,
        accumulated_stats["cumul_number_frames_played"],
    )
    tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_base_dir / (config_copy.run_name + tensorboard_suffix)))

    loss_history = []
    loss_test_history = []
    train_on_batch_duration_history = []
    grad_norm_history = []
    layer_grad_norm_history = defaultdict(list)

    # ========================================================
    # Make the trainer
    # ========================================================
    trainer = qsac.Trainer(
        policy=policy_online,
        target_policy=target_policy,
        critic1=critic1,
        critic2=critic2,
        target_critic1=target_critic1,
        target_critic2=target_critic2,
        policy_optimizer=optimizer_policy,
        critic_optimizer=optimizer_critic,
        alpha_optimizer=optimizer_alpha,
        log_alpha=log_alpha,
        scaler=scaler,
        batch_size=config_copy.batch_size,
        n_quantiles=config_copy.n_quantiles,
    )

    inferer = qsac.Inferer(
        inference_network=policy_online,
        critic=critic1,
    )

    while True:  # Trainer loop
        before_wait_time = time.perf_counter()
        wait(rollout_queue_readers)
        time_waited = time.perf_counter() - before_wait_time
        if time_waited > 1:
            print(f"Warning: learner waited {time_waited:.2f} seconds for workers to provide memories")
        time_waited_for_workers_since_last_tensorboard_write += time_waited
        for idx in queue_check_order:
            if not rollout_queues[idx].empty():
                (
                    rollout_results,
                    end_race_stats,
                    fill_buffer,
                    is_explo,
                    map_name,
                    map_status,
                    rollout_duration,
                    loop_number,
                ) = rollout_queues[idx].get()
                queue_check_order.append(queue_check_order.pop(queue_check_order.index(idx)))
                break

        importlib.reload(config_copy)

        new_tensorboard_suffix = utilities.from_staircase_schedule(
            config_copy.tensorboard_suffix_schedule,
            accumulated_stats["cumul_number_frames_played"],
        )
        if new_tensorboard_suffix != tensorboard_suffix:
            tensorboard_suffix = new_tensorboard_suffix
            tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_base_dir / (config_copy.run_name + tensorboard_suffix)))

        (
            new_memory_size,
            new_memory_size_start_learn,
        ) = utilities.from_staircase_schedule(
            config_copy.memory_size_schedule,
            accumulated_stats["cumul_number_frames_played"],
        )
        if new_memory_size != memory_size:
            buffer, buffer_test = resize_buffers(buffer, buffer_test, new_memory_size)
            offset_cumul_number_single_memories_used += (
                new_memory_size_start_learn - memory_size_start_learn
            ) * config_copy.number_times_single_memory_is_used_before_discard
            memory_size_start_learn = new_memory_size_start_learn
            memory_size = new_memory_size
        # ===============================================
        #   VERY BASIC TRAINING ANNEALING
        # ===============================================

        # LR and weight_decay calculation
        learning_rate = utilities.from_exponential_schedule(config_copy.lr_schedule, accumulated_stats["cumul_number_frames_played"])
        weight_decay = config_copy.weight_decay_lr_ratio * learning_rate
        engineered_speedslide_reward = utilities.from_linear_schedule(
            config_copy.engineered_speedslide_reward_schedule,
            accumulated_stats["cumul_number_frames_played"],
        )
        engineered_neoslide_reward = utilities.from_linear_schedule(
            config_copy.engineered_neoslide_reward_schedule,
            accumulated_stats["cumul_number_frames_played"],
        )
        engineered_kamikaze_reward = utilities.from_linear_schedule(
            config_copy.engineered_kamikaze_reward_schedule, accumulated_stats["cumul_number_frames_played"]
        )
        engineered_close_to_vcp_reward = utilities.from_linear_schedule(
            config_copy.engineered_close_to_vcp_reward_schedule, accumulated_stats["cumul_number_frames_played"]
        )
        gamma = utilities.from_linear_schedule(config_copy.gamma_schedule, accumulated_stats["cumul_number_frames_played"])

        # ===============================================
        #   RELOAD
        # ===============================================

        for optimizer, lr_ratio in (
            (optimizer_policy, config_copy.policy_lr_ratio),
            (optimizer_critic, config_copy.critic_lr_ratio),
            (optimizer_alpha, config_copy.alpha_lr_ratio),
        ):
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate * lr_ratio
                # NOTE: the key is "eps" ("epsilon" is silently ignored by torch, see optimizer constructors above)
                param_group["eps"] = config_copy.adam_epsilon
                param_group["betas"] = (config_copy.adam_beta1, config_copy.adam_beta2)

        if isinstance(buffer._sampler, PrioritizedSampler):
            buffer._sampler._alpha = config_copy.prio_alpha
            buffer._sampler._beta = config_copy.prio_beta
            buffer._sampler._eps = config_copy.prio_epsilon

        if config_copy.plot_race_time_left_curves and not is_explo and (loop_number // 5) % 17 == 0:
            race_time_left_curves(rollout_results, inferer, save_dir, map_name)
            distribution_curves(buffer, save_dir, policy_online, critic1, target_critic1, target_critic2)
            loss_distribution(buffer, save_dir, policy_online, critic1, target_critic1, target_critic2)

        accumulated_stats["cumul_number_frames_played"] += len(rollout_results["frames"])

        # ===============================================
        #   WRITE SINGLE RACE RESULTS TO TENSORBOARD
        # ===============================================
        race_stats_to_write = {
            f"race_time_ratio_{map_name}": end_race_stats["race_time_for_ratio"] / (rollout_duration * 1000),
            f"explo_race_time_{map_status}_{map_name}" if is_explo else f"eval_race_time_{map_status}_{map_name}": end_race_stats[
                "race_time"
            ]
            / 1000,
            f"explo_race_finished_{map_status}_{map_name}" if is_explo else f"eval_race_finished_{map_status}_{map_name}": end_race_stats[
                "race_finished"
            ],
            # rollout_results["q_values"] holds a per-step critic Q estimate (scalar) in the QRSAC version.
            # f"single_zone_reached_{map_status}_{map_name}": rollout_results["furthest_zone_idx"],
            "instrumentation__answer_normal_step": end_race_stats["instrumentation__answer_normal_step"],
            "instrumentation__answer_action_step": end_race_stats["instrumentation__answer_action_step"],
            "instrumentation__between_run_steps": end_race_stats["instrumentation__between_run_steps"],
            "instrumentation__grab_frame": end_race_stats["instrumentation__grab_frame"],
            "instrumentation__convert_frame": end_race_stats["instrumentation__convert_frame"],
            "instrumentation__grab_floats": end_race_stats["instrumentation__grab_floats"],
            "instrumentation__exploration_policy": end_race_stats["instrumentation__exploration_policy"],
            "instrumentation__request_inputs_and_speed": end_race_stats["instrumentation__request_inputs_and_speed"],
            "tmi_protection_cutoff": end_race_stats["tmi_protection_cutoff"],
            "worker_time_in_rollout_percentage": rollout_results["worker_time_in_rollout_percentage"],
        }
        print("Race time ratio  ", race_stats_to_write[f"race_time_ratio_{map_name}"])

        if end_race_stats["is_start_at_beginning"]:
            race_stats_to_write[f"single_zone_reached_{map_status}_{map_name}"] = rollout_results["furthest_zone_idx"]
        
        if not is_explo:
            race_stats_to_write[f"avg_Q_{map_status}_{map_name}"] = np.mean(rollout_results["q_values"])

        if end_race_stats["race_finished"] and end_race_stats["is_start_at_beginning"]:
            race_stats_to_write[f"{'explo' if is_explo else 'eval'}_race_time_finished_{map_status}_{map_name}"] = (
                end_race_stats["race_time"] / 1000
            )
            if not is_explo:
                accumulated_stats["rolling_mean_ms"][map_name] = (
                    accumulated_stats["rolling_mean_ms"].get(map_name, config_copy.cutoff_rollout_if_race_not_finished_within_duration_ms)
                    * 0.9
                    + end_race_stats["race_time"] * 0.1
                )
        if (
            (not is_explo)
            and end_race_stats["race_finished"]
            and end_race_stats["is_start_at_beginning"]
            and end_race_stats["race_time"] < 1.02 * accumulated_stats["rolling_mean_ms"][map_name]
        ):
            race_stats_to_write[f"eval_race_time_robust_{map_status}_{map_name}"] = end_race_stats["race_time"] / 1000
            if map_name in reference_times:
                for reference_time_name in ["author", "gold"]:
                    if reference_time_name in reference_times[map_name]:
                        reference_time = reference_times[map_name][reference_time_name]
                        race_stats_to_write[f"eval_ratio_{map_status}_{reference_time_name}_{map_name}"] = (
                            100 * (end_race_stats["race_time"] / 1000) / reference_time
                        )
                        race_stats_to_write[f"eval_agg_ratio_{map_status}_{reference_time_name}"] = (
                            100 * (end_race_stats["race_time"] / 1000) / reference_time
                        )

        for i in [0, 1, 2]:
            race_stats_to_write[f"action_{i}_starting_frame_{map_name}"] = end_race_stats[f"action_{i}_starting_frame"]
        race_stats_to_write[f"q_value_{0}_starting_frame_{map_name}"] = end_race_stats["value_starting_frame"]
        if not is_explo:
            for i, split_time in enumerate(
                [
                    (e - s) / 1000
                    for s, e in zip(
                        end_race_stats["cp_time_ms"][:-1],
                        end_race_stats["cp_time_ms"][1:],
                    )
                ]
            ):
                race_stats_to_write[f"split_{map_name}_{i}"] = split_time

        walltime_tb = time.time()
        for tag, value in race_stats_to_write.items():
            tensorboard_writer.add_scalar(
                tag=tag,
                scalar_value=value,
                global_step=accumulated_stats["cumul_number_frames_played"],
                walltime=walltime_tb,
            )

        # ===============================================
        #   SAVE STUFF IF THIS WAS A GOOD RACE
        # ===============================================

        if end_race_stats["is_start_at_beginning"] and end_race_stats["race_time"] < accumulated_stats["alltime_min_ms"].get(map_name, 99999999999):
            # This is a new alltime_minimum
            accumulated_stats["alltime_min_ms"][map_name] = end_race_stats["race_time"]
            if accumulated_stats["cumul_number_frames_played"] > config_copy.frames_before_save_best_runs:
                name = f"{map_name}_{end_race_stats['race_time'] // 1000}_{end_race_stats['race_time'] % 1000:03d}"
                utilities.save_run(
                    base_dir,
                    save_dir / "best_runs" / name,
                    rollout_results,
                    f"{name}.inputs",
                    inputs_only=False,
                )
                utilities.save_checkpoint(
                    save_dir / "best_runs",
                    policy_online,
                    critic1,
                    critic2,
                    target_policy,
                    target_critic1,
                    target_critic2,
                    optimizer_policy,
                    optimizer_critic,
                    log_alpha,
                    optimizer_alpha,
                    scaler,
                )

        if end_race_stats["is_start_at_beginning"] and end_race_stats["race_time"] < config_copy.threshold_to_save_all_runs_ms:
            race_time_ms = end_race_stats["race_time"]
            name = f"{map_name}_{race_time_ms // 1000}_{race_time_ms % 1000:03d}_{datetime.now().strftime('%m%d_%H%M%S')}_{accumulated_stats['cumul_number_frames_played']}_{'explo' if is_explo else 'eval'}"
            utilities.save_run(
                base_dir,
                save_dir / "good_runs",
                rollout_results,
                f"{name}.inputs",
                inputs_only=True,
            )

        # ===============================================
        #   FILL BUFFER WITH (S, A, R, S') transitions
        # ===============================================
        if fill_buffer:
            (
                buffer,
                buffer_test,
                number_memories_added_train,
                number_memories_added_test,
            ) = buffer_management.fill_buffer_from_rollout_with_n_steps_rule(
                buffer,
                buffer_test,
                rollout_results,
                config_copy.n_steps,
                gamma,
                config_copy.discard_non_greedy_actions_in_nsteps,
                engineered_speedslide_reward,
                engineered_neoslide_reward,
                engineered_kamikaze_reward,
                engineered_close_to_vcp_reward,
            )

            accumulated_stats["cumul_number_memories_generated"] += number_memories_added_train + number_memories_added_test
            shared_steps.value = accumulated_stats["cumul_number_frames_played"]
            neural_net_reset_counter += number_memories_added_train
            accumulated_stats["cumul_number_single_memories_should_have_been_used"] += (
                config_copy.number_times_single_memory_is_used_before_discard * number_memories_added_train
            )
            print(f" NMG={accumulated_stats['cumul_number_memories_generated']:<8}")

            # ===============================================
            #   PERIODIC RESET ?
            # ===============================================

            if neural_net_reset_counter >= config_copy.reset_every_n_frames_generated or single_reset_flag != config_copy.single_reset_flag:
                neural_net_reset_counter = 0
                single_reset_flag = config_copy.single_reset_flag
                accumulated_stats["cumul_number_single_memories_should_have_been_used"] += config_copy.additional_transition_after_reset

                # IQN-specific periodic soft reset machinery (A_head/V_head interpolation) was removed for QRSAC.

            # ===============================================
            #   LEARN ON BATCH
            # ===============================================

            if not policy_online.training:
                policy_online.train()

            while (
                len(buffer) >= memory_size_start_learn
                and accumulated_stats["cumul_number_single_memories_used"] + offset_cumul_number_single_memories_used
                <= accumulated_stats["cumul_number_single_memories_should_have_been_used"]
            ):
                if (random.random() < config_copy.buffer_test_ratio and len(buffer_test) > 0) or len(buffer) == 0:
                    test_start_time = time.perf_counter()
                    loss, _ = trainer.train_on_batch(buffer_test, do_learn=False)
                    time_testing_since_last_tensorboard_write += time.perf_counter() - test_start_time
                    loss_test_history.append(loss)
                    print(f"BT   {loss=:<8.2e}")
                else:
                    train_start_time = time.perf_counter()
                    loss, grad_norm = trainer.train_on_batch(buffer, do_learn=True)
                    train_on_batch_duration_history.append(time.perf_counter() - train_start_time)
                    time_training_since_last_tensorboard_write += train_on_batch_duration_history[-1]
                    accumulated_stats["cumul_number_single_memories_used"] += (
                        4 * config_copy.batch_size
                        if (len(buffer) < buffer._storage.max_size and buffer._storage.max_size > 200_000)
                        else config_copy.batch_size
                    )  # do fewer batches while memory is not full
                    loss_history.append(loss)
                    if not math.isinf(grad_norm):
                        grad_norm_history.append(grad_norm)
                        # utilities.log_gradient_norms(policy_online, layer_grad_norm_history) #~1ms overhead per batch

                    accumulated_stats["cumul_number_batches_done"] += 1
                    print(f"B    {loss=:<8.2e} {grad_norm=:<8.2e} {train_on_batch_duration_history[-1]*1000:<8.1f}")

                    utilities.custom_weight_decay(policy_online, 1 - weight_decay)
                    utilities.custom_weight_decay(critic1, 1 - weight_decay)
                    utilities.custom_weight_decay(critic2, 1 - weight_decay)
                    if accumulated_stats["cumul_number_batches_done"] % config_copy.send_shared_network_every_n_batches == 0:
                        with shared_network_lock:
                            uncompiled_shared_network.load_state_dict(uncompiled_policy_online.state_dict())

                    # ===============================================
                    #   UPDATE TARGET NETWORKS
                    # ===============================================
                    if (
                        accumulated_stats["cumul_number_single_memories_used"]
                        >= accumulated_stats["cumul_number_single_memories_used_next_target_network_update"]
                    ):
                        accumulated_stats["cumul_number_target_network_updates"] += 1
                        accumulated_stats["cumul_number_single_memories_used_next_target_network_update"] += (
                            config_copy.number_memories_trained_on_between_target_network_updates
                        )
                        # print("UPDATE")
                        utilities.soft_copy_param(target_policy, policy_online, config_copy.soft_update_tau)
                        utilities.soft_copy_param(target_critic1, critic1, config_copy.soft_update_tau)
                        utilities.soft_copy_param(target_critic2, critic2, config_copy.soft_update_tau)
            sys.stdout.flush()

        # ===============================================
        #   WRITE AGGREGATED STATISTICS TO TENSORBOARD EVERY 5 MINUTES
        # ===============================================
        save_frequency_s = 5 * 60
        if time.perf_counter() - time_last_save >= save_frequency_s:
            accumulated_stats["cumul_training_hours"] += (time.perf_counter() - time_last_save) / 3600
            time_since_last_save = time.perf_counter() - time_last_save
            waited_percentage = time_waited_for_workers_since_last_tensorboard_write / time_since_last_save
            trained_percentage = time_training_since_last_tensorboard_write / time_since_last_save
            tested_percentage = time_testing_since_last_tensorboard_write / time_since_last_save
            time_waited_for_workers_since_last_tensorboard_write = 0
            time_training_since_last_tensorboard_write = 0
            time_testing_since_last_tensorboard_write = 0
            transitions_learned_per_second = (
                accumulated_stats["cumul_number_single_memories_used"] - transitions_learned_last_save
            ) / time_since_last_save
            time_last_save = time.perf_counter()
            transitions_learned_last_save = accumulated_stats["cumul_number_single_memories_used"]

            # ===============================================
            #   COLLECT VARIOUS STATISTICS
            # ===============================================
            step_stats = {
                "gamma": gamma,
                "n_steps": config_copy.n_steps,
                "sac_alpha": trainer.alpha,
                "mean_log_pi": trainer.diagnostics["mean_log_pi"],
                "policy_loss_mean": trainer.diagnostics["policy_loss"],
                "alpha_loss_mean": trainer.diagnostics["alpha_loss"],
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "discard_non_greedy_actions_in_nsteps": config_copy.discard_non_greedy_actions_in_nsteps,
                "memory_size": len(buffer),
                "number_times_single_memory_is_used_before_discard": config_copy.number_times_single_memory_is_used_before_discard,
                "learner_percentage_waiting_for_workers": waited_percentage,
                "learner_percentage_training": trained_percentage,
                "learner_percentage_testing": tested_percentage,
                "transitions_learned_per_second": transitions_learned_per_second,
            }
            if len(loss_history) > 0 and len(loss_test_history) > 0:
                step_stats.update(
                    {
                        "loss": np.mean(loss_history),
                        "loss_test": np.mean(loss_test_history),
                        "train_on_batch_duration": np.median(train_on_batch_duration_history),
                        "grad_norm_history_q1": np.quantile(grad_norm_history, 0.25),
                        "grad_norm_history_median": np.quantile(grad_norm_history, 0.5),
                        "grad_norm_history_q3": np.quantile(grad_norm_history, 0.75),
                        "grad_norm_history_d9": np.quantile(grad_norm_history, 0.9),
                        "grad_norm_history_d98": np.quantile(grad_norm_history, 0.98),
                        "grad_norm_history_max": np.max(grad_norm_history),
                    }
                )
                for key, val in layer_grad_norm_history.items():
                    step_stats.update(
                        {
                            f"{key}_median": np.quantile(val, 0.5),
                            f"{key}_q3": np.quantile(val, 0.75),
                            f"{key}_d9": np.quantile(val, 0.9),
                            f"{key}_c98": np.quantile(val, 0.98),
                            f"{key}_max": np.max(val),
                        }
                    )
            if isinstance(buffer._sampler, PrioritizedSampler):
                all_priorities = np.array([buffer._sampler._sum_tree.at(i) for i in range(len(buffer))])
                step_stats.update(
                    {
                        "priorities_min": np.min(all_priorities),
                        "priorities_q1": np.quantile(all_priorities, 0.1),
                        "priorities_mean": np.mean(all_priorities),
                        "priorities_median": np.quantile(all_priorities, 0.5),
                        "priorities_q3": np.quantile(all_priorities, 0.75),
                        "priorities_d9": np.quantile(all_priorities, 0.9),
                        "priorities_c98": np.quantile(all_priorities, 0.98),
                        "priorities_max": np.max(all_priorities),
                    }
                )
            for key, value in accumulated_stats.items():
                if key not in ["alltime_min_ms", "rolling_mean_ms"]:
                    step_stats[key] = value
            for key, value in accumulated_stats["alltime_min_ms"].items():
                step_stats[f"alltime_min_ms_{map_name}"] = value

            loss_history = []
            loss_test_history = []
            train_on_batch_duration_history = []
            grad_norm_history = []
            layer_grad_norm_history = defaultdict(list)

            # ===============================================
            #   COLLECT POLICY DISTRIBUTION DIAGNOSTICS
            # ===============================================

            if policy_online.training:
                policy_online.eval()
            action, mu, std = inferer.infer_actions(rollout_results["frames"][0], rollout_results["state_float"][0], deterministic=True)
            for i, std_value in enumerate(std):
                step_stats[f"policy_std_dim{i}"] = std_value
            for i, mu_value in enumerate(mu):
                step_stats[f"policy_mu_dim{i}"] = mu_value
            step_stats["mean_q_starting_frame"] = inferer.estimate_q(rollout_results["frames"][0], rollout_results["state_float"][0], action)

            # ===============================================
            #   WRITE TO TENSORBOARD
            # ===============================================

            walltime_tb = time.time()
            for network_to_log, network_name in ((policy_online, "policy"), (critic1, "critic1"), (critic2, "critic2")):
                for name, param in network_to_log.named_parameters():
                    tensorboard_writer.add_scalar(
                        tag=f"layer_{network_name}_{name}_L2",
                        scalar_value=np.sqrt((param**2).mean().detach().cpu().item()),
                        global_step=accumulated_stats["cumul_number_frames_played"],
                        walltime=walltime_tb,
                    )
            try:
                for optimizer, optimizer_name, network_to_log in (
                    (optimizer_policy, "policy", policy_online),
                    (optimizer_critic, "critic1", critic1),
                ):
                    for p, (name, _) in zip(
                        optimizer.param_groups[0]["params"],
                        network_to_log.named_parameters(),
                    ):
                        state = optimizer.state[p]
                        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                        mod_lr = 1 / (exp_avg_sq.sqrt() + config_copy.adam_epsilon)
                        tensorboard_writer.add_scalar(
                            tag=f"lr_ratio_{optimizer_name}_{name}_L2",
                            scalar_value=np.sqrt((mod_lr**2).mean().detach().cpu().item()),
                            global_step=accumulated_stats["cumul_number_frames_played"],
                            walltime=walltime_tb,
                        )
                        tensorboard_writer.add_scalar(
                            tag=f"exp_avg_{optimizer_name}_{name}_L2",
                            scalar_value=np.sqrt((exp_avg**2).mean().detach().cpu().item()),
                            global_step=accumulated_stats["cumul_number_frames_played"],
                            walltime=walltime_tb,
                        )
                        tensorboard_writer.add_scalar(
                            tag=f"exp_avg_sq_{optimizer_name}_{name}_L2",
                            scalar_value=np.sqrt((exp_avg_sq**2).mean().detach().cpu().item()),
                            global_step=accumulated_stats["cumul_number_frames_played"],
                            walltime=walltime_tb,
                        )
            except:
                pass

            for k, v in step_stats.items():
                tensorboard_writer.add_scalar(
                    tag=k,
                    scalar_value=v,
                    global_step=accumulated_stats["cumul_number_frames_played"],
                    walltime=walltime_tb,
                )

            previous_alltime_min = previous_alltime_min or copy.deepcopy(accumulated_stats["alltime_min_ms"])

            tensorboard_writer.add_text(
                "times_summary",
                f"{datetime.now().strftime('%Y/%m/%d, %H:%M:%S')} "
                + " ".join(
                    [
                        f"{'**' if v < previous_alltime_min.get(k, 99999999) else ''}{k}: {v / 1000:.2f}{'**' if v < previous_alltime_min.get(k, 99999999) else ''}"
                        for k, v in accumulated_stats["alltime_min_ms"].items()
                    ]
                ),
                global_step=accumulated_stats["cumul_number_frames_played"],
                walltime=walltime_tb,
            )

            previous_alltime_min = copy.deepcopy(accumulated_stats["alltime_min_ms"])

            # ===============================================
            #   BUFFER STATS
            # ===============================================

            state_floats = np.array([experience.state_float for experience in buffer._storage])
            mean_in_buffer = state_floats.mean(axis=0)
            std_in_buffer = state_floats.std(axis=0)

            print("Raw mean in buffer  :", mean_in_buffer.round(1))
            print("Raw std in buffer   :", std_in_buffer.round(1))
            print()
            print(
                "Corr mean in buffer :",
                ((mean_in_buffer - config_copy.float_inputs_mean) / config_copy.float_inputs_std).round(1),
            )
            print(
                "Corr std in buffer  :",
                (std_in_buffer / config_copy.float_inputs_std).round(1),
            )
            print()

            # ===============================================
            #   HIGH PRIORITY TRANSITIONS
            # ===============================================
            if config_copy.make_highest_prio_figures and isinstance(buffer._sampler, PrioritizedSampler):
                highest_prio_transitions(buffer, save_dir)

            # ===============================================
            #   SAVE
            # ===============================================
            utilities.save_checkpoint(
                save_dir,
                policy_online,
                critic1,
                critic2,
                target_policy,
                target_critic1,
                target_critic2,
                optimizer_policy,
                optimizer_critic,
                log_alpha,
                optimizer_alpha,
                scaler,
            )
            joblib.dump(accumulated_stats, save_dir / "accumulated_stats.joblib")
