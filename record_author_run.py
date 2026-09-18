'''
just a timer for author to set a baseline for the ai to try to beat
'''

import time
import math
import argparse
import numpy as np

from trackmania_rl.bng_trajectory import get_vcp_centers_from_file, update_current_zone_idx
from trackmania_rl.tmi_interaction.bng_interface import VehicleInterface
from config_files import config

def main():
    parser = argparse.ArgumentParser(description="A script that greets a user.")
    parser.add_argument("-p", "--port", type=int, default=5000, help="Number of times to greet.")
    args = parser.parse_args()

    zone_centers = get_vcp_centers_from_file(trajectory_path='ecusa1.npy', config=config)
    finish_idx = len(zone_centers) - 1 - config.n_zone_centers_extrapolate_after_end_of_map
    first_zone_idx = config.n_zone_centers_extrapolate_before_start_of_map
    current_zone_idx = first_zone_idx

    target = zone_centers[finish_idx]
    radius = config.max_allowable_distance_to_virtual_checkpoint
    print(f"loaded zones from file")
    print(f"finish zone idx: {finish_idx} located at {target}")

    port = args.port
    vehicle = VehicleInterface(port)

    while True:
        state = vehicle.get_state()
        sim_time = state["sim_time"]

        position = np.asarray(state["position"], dtype=np.float32)
        current_zone_idx = update_current_zone_idx(
            current_zone_idx,
            zone_centers,
            position,
            radius,
            first_zone_idx,
            finish_idx,
        )

        if current_zone_idx >= finish_idx:
            print(f"arrived in {sim_time} sec")
            vehicle.reset()
            vehicle.step_complete()
            break

        vehicle.step_complete()


if __name__ == "__main__":
    main()