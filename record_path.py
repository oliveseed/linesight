'''
this is for creating the centerline path by manually driving and recording it
'''

import time
import math
import argparse
import numpy as np
from trackmania_rl.tmi_interaction.bng_interface import VehicleInterface

def main():
    parser = argparse.ArgumentParser(description="A script that greets a user.")
    parser.add_argument("-p", "--port", type=int, default=5000, help="Number of times to greet.")
    args = parser.parse_args()

    port = args.port
    vehicle = VehicleInterface(port)

    positions = []
    for _ in range(60*10*20):
        state = vehicle.get_state()
        sim_time = state["sim_time"]
        position = state["position"]
     
        print(sim_time, position)
        positions.append(np.array(position))

        vehicle.step_complete()
    
    path = np.stack(positions, axis=0)
    np.save("nord.npy", path)

if __name__ == "__main__":
    main()