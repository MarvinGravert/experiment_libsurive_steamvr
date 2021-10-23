"""Script to run the static accuracy experiment for steamvr and libsurvive. 

Experiment:
Follows the guidelines set out by VDI/VDE 2634:
Define mutliple poses within the desired measuring space. Measure the pose of 
both trackers at each pose.

Essentially, its simliar to the dynamic accuracy experiment but with static poses.

Setup: 
2 LH, 2 Trackers mounted on a bar facing away from each other, the bar itself 
is mounted on a robot.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


from utils.general import Framework, get_file_location, save_data, check_if_moved
from utils.GS_timing import delay
from utils.consts import (STATIC_ACC_DURATION, STATIC_ACC_FREQUENCY,
                          LIBSURVIVE_STABILITY_COUNTER, LIBSURVIVE_STABILITY_THRESHOLD)

if os.name == 'nt':  # if windows
    import openvr
    import utils.triad_openvr as triad_openvr
else:
    import pysurvive
    from utils.better_libsurvive_api import (
        BetterSurviveObject, get_n_survive_objects, get_simple_context, simple_start
    )


def run_static_accuracy_steamvr(
    frequency: int,
    duration: float,
) -> np.ndarray:
    """runs the static accuracy for steamvr. Collects data from both tracker
    and writes them into a matrix
    x y z w i j k x y z w i j k

    Args:
        frequency (int): (max) measuring frequency 
        duration (float): time during which the tracker are polled

    Returns:
        np.ndarray: nx14 matrix containing poses of both trackers
    """
    v = triad_openvr.triad_openvr()
    v.print_discovered_objects()
    counter = 0
    max_counter = duration*frequency
    pose_matrix = np.zeros((int(max_counter), 14))
    print("START Measuring")
    # time.sleep(1)
    pose_list = list()
    while counter < max_counter:
        current_time = time.perf_counter()
        pose_tracker_1 = v.devices["tracker_1"].get_pose_quaternion()
        pose_tracker_2 = v.devices["tracker_2"].get_pose_quaternion()
        pose_list.append([pose_tracker_1, pose_tracker_2])
        counter += 1
        try:
            time_2_sleep = 1/frequency-(time.perf_counter()-current_time)
            delay(time_2_sleep*1000)
        except ValueError:  # happends if negative sleep duration (loop took too long)
            pass
    for j, pose in enumerate(pose_list):
        pose_matrix[j, :] = np.array(pose[0]+pose[1])
    return pose_matrix


def run_static_accuracy_libsurvive(
    frequency: int,
    duration: float,
) -> np.ndarray:
    """runs the static accuracy for libsurvive. Collects data from both tracker
    and writes them into a matrix
    x y z w i j k x y z w i j k

    Args:
        frequency (int): (max) measuring frequency 
        duration (float): time during which the tracker are polled
    Returns:
        np.ndarray: nx14 matrix containing poses of both trackers
    """
    actx = get_simple_context(sys.argv)
    simple_start(actx)
    survive_objects = get_n_survive_objects(
        actx=actx,
        num=4
    )
    tracker_obj_1 = survive_objects["red"]
    tracker_obj_2 = survive_objects["black"]
    # run stabilizer
    last_pose = tracker_obj_2.get_pose_quaternion()
    stable_counter = 0
    time.sleep(0.05)
    print("Waiting for stability")
    while stable_counter < LIBSURVIVE_STABILITY_COUNTER:
        current_pose = tracker_obj_2.get_pose_quaternion()
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
    second_tracker_list = list()
    counter = 0
    max_counter = duration*frequency
    print("START Measuring")

    while counter < max_counter:
        current_time = time.perf_counter()
        first_tracker_list.append(tracker_obj_1.get_pose_quaternion())
        second_tracker_list.append(tracker_obj_2.get_pose_quaternion())
        counter += 1
        try:
            time_2_sleep = 1/frequency-(time.perf_counter()-current_time)
            delay(time_2_sleep*1000)
        except ValueError:  # happends if negative sleep duration (loop took too long)
            pass
    first_pose_matrix = np.array(first_tracker_list)
    second_pose_matrix = np.array(second_tracker_list)
    pose_matrix = np.hstack((first_pose_matrix, second_pose_matrix))
    return pose_matrix


if __name__ == "__main__":
    exp_num = 1
    exp_type = "static_accuracy"
    # settings:
    settings = {
        "frequency": STATIC_ACC_FREQUENCY,  # Hz
        "duration": STATIC_ACC_DURATION,  # seconds
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
        pose_matrix = run_static_accuracy_libsurvive(
            frequency=settings["frequency"],
            duration=settings["duration"]
        )
    elif framework == Framework.steamvr:
        pose_matrix = run_static_accuracy_steamvr(
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
        exp_type=exp_type,
        settings=settings,
        framework=framework
    )
    # adjusting because I deleted P3 in the .src and getting confused^^
    if num_point > 2:
        num_point += 1
    print(f"Run for point {num_point}")
    print("Finished")
