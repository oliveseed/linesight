"""
Vehicle instance manager for BNG rollouts.

One collector process owns one VehicleInterface (one vehicle). After the
interface is connected, the game is assumed ready — no window/UI/map handling.
"""

import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import numpy.typing as npt

from config_files import config_copy
from trackmania_rl.bng_trajectory import (
    make_virtual_checkpoints,
    get_vcp_centers_from_file,
    update_current_zone_idx,
)
from trackmania_rl.tmi_interaction.bng_interface import VehicleInterface
from trackmania_rl import contact_materials


class GameInstanceManager:
    """Manages one BNG vehicle connection and runs rollouts for the collector."""

    def __init__(
        self,
        game_spawning_lock,
        running_speed=1,
        run_steps_per_action=1,
        max_overall_duration_ms=2000,
        max_minirace_duration_ms=2000,
        tmi_port=None,
    ):
        self.iface = None
        self.running_speed = running_speed
        self.run_steps_per_action = run_steps_per_action
        self.max_overall_duration_ms = max_overall_duration_ms
        self.max_minirace_duration_ms = max_minirace_duration_ms
        self.tmi_port = tmi_port
        self.game_spawning_lock = game_spawning_lock
        self.last_rollout_crashed = False
        self._fake_vehicle_proc = None
        self._base_dir = Path(__file__).resolve().parents[2]

    def ensure_vehicle_connected(self):
        """Bind VehicleInterface and launch a fake vehicle client if needed."""
        if self.iface is not None:
            return

        result = {}
        error = {}

        def _accept():
            try:
                result["iface"] = VehicleInterface(self.tmi_port)
            except Exception as exc:  # noqa: BLE001 — surface to parent thread
                error["exc"] = exc

        # VehicleInterface blocks in accept(); start listener, then spawn client.
        accept_thread = threading.Thread(target=_accept, daemon=True)
        accept_thread.start()

        # Give the server socket a moment to bind+listen before the client connects.
        time.sleep(0.15)
        # self._launch_fake_vehicle()

        accept_thread.join(timeout=config_copy.tmi_protection_timeout_s)
        if accept_thread.is_alive():
            raise TimeoutError(f"Timed out waiting for vehicle on port {self.tmi_port}")
        if "exc" in error:
            raise error["exc"]
        self.iface = result["iface"]
        print(f"Vehicle connected on port {self.tmi_port}")

    def _launch_fake_vehicle(self):
        """Start bng_test_fake_vehicle.py as the stand-in for a real BNG vehicle."""
        if self._fake_vehicle_proc is not None and self._fake_vehicle_proc.poll() is None:
            return
        fake_vehicle_script = self._base_dir / "bng_test_fake_vehicle.py"
        with self.game_spawning_lock:
            self._fake_vehicle_proc = subprocess.Popen(
                [sys.executable, str(fake_vehicle_script), "-p", str(self.tmi_port)],
                cwd=str(self._base_dir),
            )

    def request_inputs(self, action: npt.NDArray):
        """Send a continuous (steer, throttle, brake) action to the vehicle."""
        self.iface.send_action(
            steer=float(np.clip(action[0], -1.0, 1.0)),
            throttle=float(np.clip(action[1], 0.0, 1.0)),
            brake=float(np.clip(action[2], 0.0, 1.0)),
        )

    def _blank_frame(self) -> npt.NDArray:
        # Placeholder image so the conv head stays in the graph for later re-use.
        return np.zeros((1, config_copy.H_downsized, config_copy.W_downsized), dtype=np.uint8)

    def _orientation_matrix(self, state: Dict) -> npt.NDArray:
        """World-to-car rotation: rows are right, up, forward in world coords.

        BNG is z-up. Columns of R map car-frame basis to world; R.T maps world→car.
        """
        right = np.asarray(state["right_vector"], dtype=np.float32)
        up = np.asarray(state["up_vector"], dtype=np.float32)
        forward = np.asarray(state["forward_vector"], dtype=np.float32)
        return np.column_stack((right, up, forward)).astype(np.float32)

    def _reset_to_zone(self, zone_centers, zone_idx):
        target_zone, next_zone = zone_centers[zone_idx], zone_centers[zone_idx + 1]
        d = next_zone - target_zone
        d /= np.linalg.norm(d)
        dx, dy, dz = d
        angle = -np.arctan2(dx, -dy)
        self.iface.reset_and_teleport(list(target_zone), [0, 0, angle])

    def _build_float_inputs(
        self,
        state: dict,
        current_zone_idx: int,
        zone_centers: npt.NDArray,
        distance_since_track_begin: float,
        distance_from_start_track_to_prev_zone_transition: npt.NDArray,
        actions: list,
    ) -> npt.NDArray:
        """Build the float feature vector.

        Layout (continuous-action QRSAC version), mirrors game_instance_manager:
            [0]  mini-race counter | [1..15] prev (steer,throttle,brake) x5 | [16..47] gear/wheels
            [48..50] angvel car | [51..53] velocity car | [54..56] map-up car | [57..176] 40 vcp
            [177] finish distance | [178] is_freewheeling
        """
        position = np.asarray(state["position"], dtype=np.float32)
        velocity = np.asarray(state["velocity"], dtype=np.float32)
        angular_velocity = np.asarray(state["angular_velocity"], dtype=np.float32)
        R = self._orientation_matrix(state)  # car→world
        R_inv = R.T  # world→car

        state_zone_center_coordinates_in_car_reference_system = R_inv.dot(
            (
                zone_centers[
                    current_zone_idx : current_zone_idx
                    + config_copy.one_every_n_zone_centers_in_inputs
                    * config_copy.n_zone_centers_in_inputs : config_copy.one_every_n_zone_centers_in_inputs,
                    :,
                ]
                - position
            ).T
        ).T  # (n_zone_centers_in_inputs, 3)

        # World up (z-up) expressed in the car frame — analogous to TM's y-up map vector.
        state_y_map_vector_in_car_reference_system = R_inv.dot(np.array([0.0, 0.0, 1.0], dtype=np.float32))
        state_car_velocity_in_car_reference_system = R_inv.dot(velocity)
        state_car_angular_velocity_in_car_reference_system = R_inv.dot(angular_velocity)

        # list containing n_prev_actions_in_inputs np arrays, each being a (steer, throttle, brake) action.
        previous_actions = [
            actions[k] if k >= 0 and k < len(actions) and isinstance(actions[k], np.ndarray) else config_copy.default_action
            for k in range(len(actions) - config_copy.n_prev_actions_in_inputs, len(actions))
        ]

        # Map available BNG signals into the gear/wheels block; pad the rest.
        # [4 sliding, 4 ground contact, 4 damper, gearbox, gear, rpm, counter, 16 contact materials]
        # sliding = (slip > 0).astype(np.float32)
        # ground_contact = np.ones(4, dtype=np.float32)  # placeholder: always in contact
        # damper = np.zeros(4, dtype=np.float32)
        # contact_materials = np.zeros(
        #     4 * config_copy.n_contact_material_physics_behavior_types, dtype=np.float32
        # )
        slip = np.asarray(state["slips"], dtype=np.float32)
        downforce = np.asarray(state["downforces"], dtype=np.float32) / (1536 * 9.81)  # hard coded car weight
        wheel_contact_types = np.asarray(
            [
                i == contact_materials.physics_behavior_fromint[whl]
                for whl in state["materials"]
                for i in range(config_copy.n_contact_material_physics_behavior_types)
            ],
            dtype=np.float32,
        )

        sim_state_car_gear_and_wheels = np.concatenate(
            [
                np.array(
                    [
                        float(state["throttle"]),
                        float(state["brake"]),
                        float(state["steering"]),
                        0.0, # make the vehicle believe that it is not damanged
                    ],
                    dtype=np.float32,
                ),
                slip,
                # ground_contact,
                downforce,
                np.array(
                    [
                        0.0,  # gearbox_state placeholder
                        float(state["gear"]),
                        float(state["rpm"]),
                        0.0,  # gearbox counter placeholder
                    ],
                    dtype=np.float32,
                ),
                wheel_contact_types,
            ]
        )

        finish_distance = distance_from_start_track_to_prev_zone_transition[
            len(zone_centers) - config_copy.n_zone_centers_extrapolate_after_end_of_map - 1
            if len(distance_from_start_track_to_prev_zone_transition)
            > len(zone_centers) - config_copy.n_zone_centers_extrapolate_after_end_of_map - 1
            else -1
        ]
        # The distance array is indexed by zone transition; use finish_idx-aligned value.
        finish_zone = len(zone_centers) - config_copy.n_zone_centers_extrapolate_after_end_of_map
        finish_distance = distance_from_start_track_to_prev_zone_transition[
            min(finish_zone - 1, len(distance_from_start_track_to_prev_zone_transition) - 1)
        ]

        is_freewheeling = 1.0 if state["throttle"] <= 0.0 and state["brake"] <= 0.0 else 0.0

        floats = np.hstack(
            (
                0,  # temporal mini-race feature, filled later in buffer_collate_function
                np.array(
                    [
                        previous_action[input_idx]
                        for previous_action in previous_actions
                        for input_idx in range(3)  # steer, throttle, brake
                    ],
                    dtype=np.float32,
                ),
                sim_state_car_gear_and_wheels.ravel(),
                state_car_angular_velocity_in_car_reference_system.ravel(),
                state_car_velocity_in_car_reference_system.ravel(),
                state_y_map_vector_in_car_reference_system.ravel(),
                state_zone_center_coordinates_in_car_reference_system.ravel(),
                min(
                    config_copy.margin_to_announce_finish_meters,
                    finish_distance - distance_since_track_begin,
                ),
                is_freewheeling,
            )
        ).astype(np.float32)
        return floats

    def _meters_along_centerline(
        self,
        position: npt.NDArray,
        current_zone_idx: int,
        zone_transitions: npt.NDArray,
        distance_between_zone_transitions: npt.NDArray,
        distance_from_start_track_to_prev_zone_transition: npt.NDArray,
        normalized_vector_along_track_axis: npt.NDArray,
    ) -> float:
        seg_idx = current_zone_idx - 1
        seg_idx = int(np.clip(seg_idx, 0, len(distance_between_zone_transitions) - 1))
        meters_in_current_zone = np.clip(
            (position - zone_transitions[seg_idx]).dot(normalized_vector_along_track_axis[seg_idx]),
            0,
            distance_between_zone_transitions[seg_idx],
        )
        return float(distance_from_start_track_to_prev_zone_transition[seg_idx] + meters_in_current_zone)

    def rollout(
        self,
        exploration_policy: Callable,
        zone_centers: npt.NDArray,
    ):
        # zone_centers = get_vcp_centers_from_file(trajectory_path=None, config=config_copy)
        finish_idx = len(zone_centers) - 1 - config_copy.n_zone_centers_extrapolate_after_end_of_map
        first_zone_idx = config_copy.n_zone_centers_extrapolate_before_start_of_map
        vcp = make_virtual_checkpoints(zone_centers, finish_idx)
        zone_transitions = vcp.zone_transitions
        distance_between_zone_transitions = vcp.distance_between_zone_transitions
        distance_from_start_track_to_prev_zone_transition = vcp.distance_from_start_track_to_prev_zone_transition
        normalized_vector_along_track_axis = vcp.normalized_vector_along_track_axis

        self.ensure_vehicle_connected()

        actions = []

        # start_zone_idx = first_zone_idx
        start_zone_idx = np.random.randint(first_zone_idx + 17500, first_zone_idx + 18000)

        self.last_rollout_crashed = False
        current_zone_idx = start_zone_idx  # first_zone_idx
        last_progress_improvement_ms = 0
        n_th_action_we_compute = 0
        race_time_ms = 0
        furthest_zone_idx = 0

        try:
            # Reset puts the vehicle at the start and clears its timers/state.
            # self.iface.reset()
            self._reset_to_zone(zone_centers, start_zone_idx)
            self.iface.step_complete()

            # TODO: fix the reset bug to avoid having to do this
            _ = self.iface.get_state()
            self.iface.step_complete()

            state = self.iface.get_state()
            race_time_ms = int(round(float(state["sim_time"]) * 1000))

            # print("initial state", state)

            this_rollout_is_finished = False
            while not this_rollout_is_finished:
                position = np.asarray(state["position"], dtype=np.float32)
                current_zone_idx = update_current_zone_idx(
                    current_zone_idx,
                    zone_centers,
                    position,
                    config_copy.max_allowable_distance_to_virtual_checkpoint,
                    start_zone_idx,  # first_zone_idx,
                    finish_idx,
                )

                # print("progress", current_zone_idx, rollout_results['furthest_zone_idx'], race_time_ms)
                if current_zone_idx > furthest_zone_idx:
                    last_progress_improvement_ms = race_time_ms
                    furthest_zone_idx = current_zone_idx

                distance_since_track_begin = self._meters_along_centerline(
                    position,
                    current_zone_idx,
                    zone_transitions,
                    distance_between_zone_transitions,
                    distance_from_start_track_to_prev_zone_transition,
                    normalized_vector_along_track_axis,
                )

                wheels_off_track = [whl in contact_materials.forbidden_materials for whl in state["materials"]]

                floats = self._build_float_inputs(
                    state,
                    current_zone_idx,
                    zone_centers,
                    distance_since_track_begin,
                    distance_from_start_track_to_prev_zone_transition,
                    actions,
                )

                # frame = self._blank_frame()
                frame = self.iface.get_visual_input(vis_type="color")
                # print(frame)
                # TODO: check if first frame of rollout is black

                # Finished the race (reached final real VCP).
                if current_zone_idx >= finish_idx:
                    print(f"[TRUNCATION] vehicle {self.tmi_port} reached the finish")
                    this_rollout_is_finished = True
                    break

                # Failed to finish in time / no progress.
                # if (
                #     race_time_ms > self.max_overall_duration_ms
                #     or race_time_ms > last_progress_improvement_ms + self.max_minirace_duration_ms
                # ):
                #     this_rollout_is_finished = True
                #     print(f"[TRUNCATION] vehicle {self.tmi_port} timed out at zone {current_zone_idx}")
                #     break

                (
                    action,
                    action_was_greedy,
                    q_value,
                    _,
                ) = exploration_policy(frame, floats)

                actions.append(action)

                # Vehicle got too damaged
                # if state["damage_overall"] >= config_copy.max_damage_ceiling:
                #     print(f"[TRUNCATION] vehicle {self.tmi_port} crashed at zone {current_zone_idx} {state['damage_overall']}")
                #     this_rollout_is_finished = True
                #     break

                self.request_inputs(action)
                # # Apply the same action for run_steps_per_action physics ticks.
                # for _ in range(self.run_steps_per_action):
                self.iface.step_complete()
                state = self.iface.get_state()

                race_time_ms = int(round(float(state["sim_time"]) * 1000))
                n_th_action_we_compute += 1

                # print("state", state)
            # print("time", race_time_ms, last_progress_improvement_ms, self.max_minirace_duration_ms)
            # print("rollout", rollout_results['current_zone_idx'], rollout_results['furthest_zone_idx'], rollout_results['actions'])
            # print("furthest vcp reached", rollout_results['furthest_zone_idx'])
            # print("damages", rollout_results['damages'])

        except (ConnectionError, OSError, TimeoutError) as err:  # (ConnectionError, OSError, TimeoutError) as err:
            print("Cutoff rollout due to vehicle interface error:", err)
            self.last_rollout_crashed = True
            self.iface = None

        return None
