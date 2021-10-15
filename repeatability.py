import os
import sys
import time
from pathlib import Path

from utils.general import Framework, get_file_location, save_data, check_if_moved
import numpy as np
from better_libsurvive_api import (
    BetterSurviveObject, get_n_survive_objects, get_simple_context, simple_start
)
from GS_timing import delay

if os.name == 'nt':  # if windows
    import openvr
    import utils.triad_openvr as triad_openvr
else:
    import pysurvive


def run_libsurvive_repeatability(
    frequency: float,
    duration: float
) -> np.ndarray:

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
    while stable_counter < 10:
        current_pose = tracker_obj_1.get_pose_quaternion()
        if not check_if_moved(
            initial_pose=last_pose,
            current_pose=current_pose,
            moving_threshold=0.001
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


def run_repeatability_steamvr(
    frequency: int = 10,
    duration: float = 10*60
) -> np.ndarray:
    """

    Args:
        frequency (int, optional): [description]. Defaults to 10.
        duration (float, optional): [description]. Defaults to 10*60.

    Returns:
        np.ndarray: duration*frequencyx7 pose_matrix
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
            delay(time_2_sleep)
        except ValueError:  # happends if negative sleep duration (loop took too long)
            pass
    pose_matrix = np.array(pose_list)
    return pose_matrix


if __name__ == "__main__":
    point_number = 10  # 1-10
    exp_type = "repeatability"
    # settings:
    settings = {
        "frequency": 150,  # Hz
        "duration": 2,  # seconds
        "sys.args": sys.argv
    }
    framework = Framework("libsurvive")

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
    # run program
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
