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
    vehicle.reset()
    vehicle.step_complete()

    target = (0, 20, 0)
    radius = 5

    asdf = 0
    while True:
        asdf += 1
        state = vehicle.get_state()
        sim_time = state["sim_time"]
        position = state["position"]

        # print(state["throttle"])
        
        # distance = math.dist((target[0], target[1]), (position[0], position[1]))
        # if distance < radius:
        #     print("arrived")
        #     vehicle.reset()
        #     vehicle.step_complete()
        #     continue

        left = False
        right = False
        throttle = True
        brake = False
        reset_throttle = False
        reset_brake = False
        
        # if asdf >= 4:# and asdf <100:
        #     left = False
        #     throttle = False

        #time.sleep(0.25)
        vehicle.send_action(left, right, throttle, brake, reset_throttle, reset_brake)
        vehicle.step_complete()

if __name__ == "__main__":
    main()