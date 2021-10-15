import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from better_libsurvive_api import (
    BetterSurviveObject, get_n_survive_objects, get_simple_context, simple_start
)
from utils.general import Framework, get_file_location, save_data, check_if_moved
from utils.linear_algebrea_helper import (
    distance_between_hom_matrices,
    transform_to_homogenous_matrix
)
from GS_timing import delay

if os.name == 'nt':  # if windows
    import openvr
    import utils.triad_openvr as triad_openvr
else:
    import pysurvive


def run_dynamic_accuarcy_steamvr(
    frequency: int,
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
    """
    WAIT FOR tracker to move before starting measuring
    """
    print("WAITING for tracker being moved")
    inital_pose = v.devices["tracker_1"].get_pose_quaternion()
    while not check_if_moved(
        current_pose=v.devices["tracker_1"].get_pose_quaternion(),
        initial_pose=inital_pose,
        moving_threshold=0.02
    ):
        time.sleep(0.1)
    print("START Measuring")
    starttime = time.perf_counter()
    first_tracker_list = list()
    second_tracker_list = list()
    interval = 1/frequency
    try:
        while True:
            current_time = time.perf_counter()
            try:
                first_tracker_list.append(v.devices["tracker_1"].get_pose_quaternion())
            except ZeroDivisionError:  # happends if tracking is lost
                continue
            try:
                second_tracker_list.append(v.devices["tracker_2"].get_pose_quaternion())
            except ZeroDivisionError:
                del first_tracker_list[-1]
            counter += 1  # counter before adding because it may otherwise cause a matrix limit expection later on in case of interrupt
            time_2_sleep = interval-(time.perf_counter()-current_time)
            try:
                delay(time_2_sleep*1000)
            except ValueError:  # happends if negative sleep duration (loop took too long)
                pass
    except KeyboardInterrupt:
        endtime = time.perf_counter()
        duration = endtime-starttime
        actual_frequency = counter/duration
        print(
            f"Stopped measuring. After {counter} meausrements in roughly {duration} seconds. Resulting in a frequency of {actual_frequency}")
    settings["duration"] = duration
    settings["measurements"] = counter
    settings["actual frequency"] = actual_frequency

    first_pose_matrix = np.array(first_tracker_list)
    second_pose_matrix = np.array(second_tracker_list)
    if len(first_pose_matrix) != len(second_pose_matrix):
        # happends if untimely interrupt thus we cut the last entry in for the first
        first_pose_matrix = first_pose_matrix[:-1, :]
    print(first_pose_matrix.shape)
    print(second_pose_matrix.shape)
    pose_matrix = np.hstack((first_pose_matrix, second_pose_matrix))
    return pose_matrix


def run_dynamic_accuarcy_libsurvive(
    frequency: int,
) -> np.ndarray:
    actx = get_simple_context(sys.argv)
    simple_start(actx)
    survive_objects = get_n_survive_objects(
        actx=actx,
        num=4
    )
    time.sleep(1)
    tracker_obj_1 = survive_objects["red"]
    tracker_obj_2 = survive_objects["black"]
    # run stabilizer
    last_pose = tracker_obj_2.get_pose_quaternion()
    stable_counter = 0
    time.sleep(0.05)
    print("Waiting for stability")
    while stable_counter < 10:
        current_pose = tracker_obj_2.get_pose_quaternion()
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
    second_tracker_list = list()
    counter = 0
    interval = 1/frequency
    initial_pose = tracker_obj_2.get_pose_quaternion()
    while not check_if_moved(
        initial_pose=initial_pose,
        current_pose=tracker_obj_2.get_pose_quaternion(),
        moving_threshold=0.02
    ):
        time.sleep(0.01)
    print("START Measuring")
    starttime = time.perf_counter()
    while True:
        current_time = time.perf_counter()

        first_tracker_list.append(tracker_obj_1.get_pose_quaternion())
        second_tracker_list.append(tracker_obj_2.get_pose_quaternion())
        counter += 1
        try:
            time_2_sleep = interval-(time.perf_counter()-current_time)
            delay(time_2_sleep*1000)
        except ValueError:  # happends if negative sleep duration (loop took too long)
            pass
        except KeyboardInterrupt:
            endtime = time.perf_counter()
            duration = endtime-starttime
            actual_frequency = counter/duration
            print(
                f"Stopped measuring. After {counter} meausrements in roughly {duration} seconds. Resulting in a frequency of {actual_frequency}")
            break

    settings["duration"] = duration
    settings["measurements"] = counter
    settings["actual frequency"] = actual_frequency

    first_pose_matrix = np.array(first_tracker_list)
    second_pose_matrix = np.array(second_tracker_list)

    if len(first_pose_matrix) != len(second_pose_matrix):
        # happends if untimely interrupt thus we cut the last entry in for the first
        first_pose_matrix = first_pose_matrix[:-1, :]

    print(first_pose_matrix.shape)
    print(second_pose_matrix.shape)
    pose_matrix = np.hstack((first_pose_matrix, second_pose_matrix))
    return pose_matrix


if __name__ == "__main__":
    exp_num = 4
    exp_type = "dynamic_accuracy"
    # settings:
    settings = {
        "frequency": 150,  # Hz
        "velocity": "1000 mm/s",
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
    """
    RUN PROGRAM
    """
    if framework == Framework.libsurvive:
        pose_matrix = run_dynamic_accuarcy_libsurvive(
            frequency=settings["frequency"],
        )
    elif framework == Framework.steamvr:
        pose_matrix = run_dynamic_accuarcy_steamvr(
            frequency=settings["frequency"],
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
