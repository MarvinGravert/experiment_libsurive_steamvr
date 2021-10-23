"""File to run the repeatability experiment for libsurvive and steamvr.

Experiment: 
10 position inside the robot workspace are chosen. The tracker is brought to each consecutively. A measurement is
taken using this script. This is repeated 10 times. This procedure should result in 100 measurement files.

Setup: 2 LHs and 1 Tracker mounted on a robot

The duration and sampling frequency can be adjusted. I have chosen 2 seconds at 150Hz. Note: for libsurvive a waiting
period to stabilize the pose measurements precedes the actual recording period.

To remain consistent with how the data is stored, this script requires more manual input than the other scripts.
The number of each point has to manually entered before measurement are taken to ensure the data is stored correctly.
Moreover, the range of points considered for analysis have to be specified. In short this is not ideal and should 
be amended.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

from utils.GS_timing import delay
from utils.general import Framework, get_file_location, save_data, check_if_moved
from utils.consts import (
    REPEATABILITY_DURATION, REPEATABILITY_FREQUENCY, LIBSURVIVE_STABILITY_COUNTER,
    LIBSURVIVE_STABILITY_THRESHOLD
)

if os.name == 'nt':  # if windows
    import openvr
    import utils.triad_openvr as triad_openvr
else:
    import pysurvive
    from utils.better_libsurvive_api import (
        BetterSurviveObject, get_n_survive_objects, get_simple_context, simple_start
    )


def run_libsurvive_repeatability(
    frequency: float,
    duration: float
) -> np.ndarray:

    actx = get_simple_context(sys.argv)
    simple_start(actx)  # start the thread running libsurvive
    survive_objects = get_n_survive_objects(
        actx=actx,
        num=3  # LH+1 Tracker.
    )

    tracker_obj_1 = survive_objects["red"]
    """
    Stabilize the sensor fusion
    """
    last_pose = tracker_obj_1.get_pose_quaternion()
    stable_counter = 0
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
    """
    Stabilize the sensor fusion
    """
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
            delay(time_2_sleep*1000)  # function wants mili seconds
        except ValueError:  # happends if negative sleep duration (loop took too long)
            pass
    first_pose_matrix = np.array(first_tracker_list)
    return first_pose_matrix


def run_repeatability_steamvr(
    frequency: int,
    duration: float
) -> np.ndarray:
    """ Runs the SteamVR routine to gather pose data. Returns matrix Nx7 matrix
    containing the poses in x y z w i j k
    Args:
        frequency (int, optional): 
        duration (float, optional):

    Returns:
        np.ndarray: matrix with dimension (duration*frequency) x 7
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
            delay(time_2_sleep*1000)  # function wants mili seconds
        except ValueError:  # happends if negative sleep duration (loop took too long)
            pass
    pose_matrix = np.array(pose_list)
    return pose_matrix


if __name__ == "__main__":
    """ 
    SPECIFY EXPERIMENT SETTINGS HERE
    """
    # 1-10 #CHANGE to the point intended to be measured.
    point_number = 10
    exp_type = "repeatability"
    # settings:
    settings = {
        "frequency": REPEATABILITY_FREQUENCY,  # Hz
        "duration": REPEATABILITY_DURATION,  # seconds
        "sys.args": sys.argv
    }
    framework = Framework("steamvr")
    """
    CREATE NEW FILE LOCATION
    increase point number until file doesnt yet exist
    """
    num_point = 1
    file_location: Path = get_file_location(
        exp_type=exp_type,
        exp_num=point_number,
        framework=framework,
        num_point=num_point
    )
    while file_location.exists():
        num_point += 1
        file_location: Path = get_file_location(
            exp_type=exp_type,
            exp_num=point_number,
            framework=framework,
            num_point=num_point
        )
    """
    RUN PROGRAM
    """
    if framework == Framework.libsurvive:
        pose_matrix = run_libsurvive_repeatability(
            frequency=settings["frequency"],
            duration=settings["duration"]
        )
    elif framework == Framework.steamvr:
        pose_matrix = run_repeatability_steamvr(
            frequency=settings["frequency"],
            duration=settings["duration"],
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
