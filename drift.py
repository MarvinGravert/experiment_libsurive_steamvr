"""Runs script for the drift experiment.

Experiment
Measure the pose of the tracker for xx amount of time with a defined frequency


Setup: 
2 LHs with 1 Tracker mounted on a robot
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

from utils.general import Framework, save_data, get_file_location, check_if_moved
from utils.GS_timing import delay
from utils.consts import (
    DRIFT_DURATION, DRIFT_FREQUENCY, LIBSURVIVE_STABILITY_COUNTER, LIBSURVIVE_STABILITY_THRESHOLD
)

if os.name == 'nt':
    import openvr
    import utils.triad_openvr as triad_openvr
else:
    import pysurvive
    from utils.better_libsurvive_api import (
        BetterSurviveObject, get_n_survive_objects, get_simple_context, simple_start
    )


def run_drift_steamvr(
    frequency: int,
    duration: float
) -> np.ndarray:
    """
    Args:
        frequency (int): 
        duration (float): 

    Returns:
        np.ndarray: (duration*frequency) x 7 pose_matrix
    """
    v = triad_openvr.triad_openvr()
    v.print_discovered_objects()
    counter = 0
    max_counter = duration*frequency
    print("START Measuring")
    time.sleep(1)
    pose_list = list()
    while counter < max_counter:
        current_time = time.perf_counter()
        pose = v.devices["tracker_1"].get_pose_quaternion()
        pose_list.append(pose)
        counter += 1
        try:
            time_2_sleep = 1/frequency-(time.perf_counter()-current_time)
            delay(time_2_sleep*1000)
        except ValueError:  # happends if negative sleep duration (loop took too long)
            pass
    pose_matrix = np.array(pose_list)
    return pose_matrix


def run_drift_libsurvive(
    frequency: int,
    duration: float
) -> np.ndarray:
    """runs the routine to get the tracker pose from libsurive for the given duration
    in seconds with the given frequency

    Args:
        frequency (int): 
        duratation (float): 

    Returns:
        np.ndarray: (duration*frequency) x 7 pose matrix
    """
    actx = get_simple_context(sys.argv)
    simple_start(actx)
    survive_objects = get_n_survive_objects(
        actx=actx,
        num=3
    )
    tracker_obj_1 = survive_objects["red"]
    # run stabilizer
    last_pose = tracker_obj_1.get_pose_quaternion()
    stable_counter = 0
    time.sleep(0.05)
    print("Waiting for stability")
    while stable_counter < LIBSURVIVE_STABILITY_COUNTER:
        current_pose = tracker_obj_1.get_pose_quaternion()
        if not check_if_moved(
            initial_pose=last_pose,
            current_pose=current_pose,
            moving_threshold=LIBSURVIVE_STABILITY_THRESHOLD
        ):
            stable_counter += 1

        last_pose = current_pose
        time.sleep(0.1)
    print("Stable")
    first_tracker_list = list()
    counter = 0
    max_counter = duration*frequency
    print("START Measuring")

    while counter < max_counter:
        current_time = time.perf_counter()
        first_tracker_list.append(tracker_obj_1.get_pose_quaternion())
        counter += 1
        try:
            time_2_sleep = 1/frequency-(time.perf_counter()-current_time)
            delay(time_2_sleep*1000)
        except ValueError:  # happends if negative sleep duration (loop took too long)
            pass
    first_pose_matrix = np.array(first_tracker_list)
    return first_pose_matrix


if __name__ == "__main__":
    """ 
    SPECIFY EXPERIMENT SETTINGS HERE
    """
    exp_num = 1
    num_point = 1  # Point measured
    exp_type = "drift"
    settings = {
        "frequency": DRIFT_FREQUENCY,  # Hz
        "duration": DRIFT_DURATION,  # seconds
        "sys.args": sys.argv
    }
    framework = Framework("steamvr")
    """
    CREATE NEW FILE LOCATION
    increase point number until file doesnt yet exist
    """
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
    """
    RUN PROGRAM
    """
    if framework == Framework.libsurvive:
        pose_matrix = run_drift_libsurvive(
            frequency=settings["frequency"],
            duration=settings["duration"]
        )
    elif framework == Framework.steamvr:
        pose_matrix = run_drift_steamvr(
            frequency=settings["frequency"],
            duration=settings["duration"]
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
        settings=settings,
        framework=framework,
        exp_type=exp_type
    )
    print("Done with Drift")
