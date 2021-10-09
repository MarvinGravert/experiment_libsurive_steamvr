import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import extract

from utils.general import Framework, get_file_location, save_data

if os.name == 'nt':  # if windows
    import openvr
    import utils.triad_openvr as triad_openvr
else:
    import pysurvive


def run_static_accuracy_steamvr(
    frequency: int,
    duration: float,
    num_point: int
) -> np.ndarray:
    """runs the static accuracy for steamvr. Collects data from both tracker
    and writes them into a matrix
    x y z w i j k x y z w i j k

    Args:
        frequency (int, optional): [description]. Defaults to 50.
        duration (float, optional): [description]. Defaults to 10.

    Returns:
        np.ndarray: [description]
    """
    v = triad_openvr.triad_openvr()
    v.print_discovered_objects()
    counter = 0
    max_counter = duration*frequency
    pose_matrix = np.zeros((int(max_counter), 14))
    print("START Measuring")
    time.sleep(1)
    pose_list = list()
    while counter < max_counter:
        current_time = time.perf_counter()
        pose_tracker_1 = v.devices["tracker_1"].get_pose_quaternion()
        pose_tracker_2 = v.devices["tracker_2"].get_pose_quaternion()
        pose_list.append([pose_tracker_1, pose_tracker_2])
        try:
            time_2_sleep = 1/frequency-(time.perf_counter()-current_time)
            time.sleep(time_2_sleep)
        except ValueError:  # happends if negative sleep duration (loop took too long)
            pass
        counter += 1
    for j, pose in enumerate(pose_list):
        pose_matrix[j, :] = np.array(pose[0]+pose[1])
    return pose_matrix


if __name__ == "__main__":
    exp_num = 1
    exp_type = "static_accuracy"
    # settings:
    settings = {
        "frequency": 100,  # Hz
        "duration": 1  # seconds
    }
    framework = Framework("steamvr")

    """
    CREATE NEW FILE LOCATION
    increase point number until file doesnt yet exist
    """
    num_point = 0
    file_location: Path = get_file_location(
        exp_type=exp_type,
        exp_num=exp_num,
        framework=framework,
        num_point=num_point
    )
    while file_location.exists():
        num_point += 1
        file_location: Path = get_file_location(
            exp_type=exp_type,
            exp_num=exp_num,
            framework=framework,
            num_point=num_point
        )
    # run program
    """
    RUN PROGRAM
    """
    if framework == Framework.libsurvive:
        pose_matrix = run_drift_libsurvive(
            frequency=settings["frequency"],
            duration=settings["duration"]
        )
    elif framework == Framework.steamvr:
        pose_matrix = run_static_accuracy_steamvr(
            frequency=settings["frequency"],
            duration=settings["duration"],
            num_point=num_point
        )
    else:
        print("framework not recognized")
        exit()
    """ 
        ---------
        SAVE DATA
        ---------
    """
    save_data(
        file_location=file_location,
        pose_matrix=pose_matrix,
        exp_type=exp_type,
        settings=settings,
        framework=framework
    )
