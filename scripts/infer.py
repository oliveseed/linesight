# =======================================================================================================================
# Handle configuration file before running the actual content of infer.py
# =======================================================================================================================
"""
Two files named "config.py" and "config_copy.py" coexist in the same folder.

At the beginning of inference, parameters are copied from config.py to config_copy.py.
"""

import shutil
from pathlib import Path


def copy_configuration_file():
    base_dir = Path(__file__).resolve().parents[1]
    shutil.copyfile(
        base_dir / "config_files" / "config.py",
        base_dir / "config_files" / "config_copy.py",
    )


if __name__ == "__main__":
    copy_configuration_file()

# =======================================================================================================================
# Actual start of infer.py, after copying config.py
# =======================================================================================================================

import os
import signal
import sys

import torch
import torch.multiprocessing as mp
from art import tprint

from config_files import config_copy
from trackmania_rl.multiprocess.inference_process import inference_process_fn
from trackmania_rl.utilities import set_random_seed

# noinspection PyUnresolvedReferences
torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)
torch.set_float32_matmul_precision("high")
random_seed = 444
set_random_seed(random_seed)


def signal_handler(sig, frame):
    print("Received SIGINT signal. Killing all open Trackmania instances.")
    clear_tm_instances()

    for child in mp.active_children():
        child.kill()

    tprint("Bye bye!", font="tarty1")
    sys.exit()


def clear_tm_instances():
    if config_copy.is_linux:
        os.system("pkill -9 TmForever.exe")
    else:
        os.system("taskkill /F /IM TmForever.exe")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    base_dir = Path(__file__).resolve().parents[1]
    save_dir = base_dir / "save" / config_copy.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Run:\n\n")
    tprint(config_copy.run_name, font="tarty4")
    print("\n" * 2)
    tprint("Linesight", font="tarty1")
    print("\n" * 2)
    print("Inference is starting!")

    game_spawning_lock = mp.Lock()

    collector_processes = [
        mp.Process(
            target=inference_process_fn,
            args=(
                game_spawning_lock,
                base_dir,
                save_dir,
                config_copy.base_tmi_port + process_number,
                process_number,
            ),
        )
        for process_number in range(8)#range(config_copy.gpu_collectors_count)
    ]
    for collector_process in collector_processes:
        collector_process.start()

    for collector_process in collector_processes:
        collector_process.join()
