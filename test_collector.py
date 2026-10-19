import argparse

from trackmania_rl.tmi_interaction.bng_interface import VehicleInterface


def main():
    parser = argparse.ArgumentParser(description="A script that greets a user.")
    parser.add_argument("-p", "--port", type=int, default=5000, help="Number of times to greet.")
    args = parser.parse_args()

    port = args.port

    vehicle = VehicleInterface(port)
    vehicle.reset()
    vehicle.step_complete()

    asdf = 0
    while True:
        asdf += 1
        state = vehicle.get_state()
        state["sim_time"]
        state["position"]

        # print(state["throttle"])

        # distance = math.dist((target[0], target[1]), (position[0], position[1]))
        # if distance < radius:
        #     print("arrived")
        #     vehicle.reset()
        #     vehicle.step_complete()
        #     continue

        # time.sleep(0.25)
        vehicle.send_action(0.0, 1.0, 0.0)
        vehicle.step_complete()


if __name__ == "__main__":
    main()
