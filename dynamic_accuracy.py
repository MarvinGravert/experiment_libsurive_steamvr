"""Script to run the dynamic accuracy experiment for steamvr and libsurvive. 

Experiment:
Follows the guidelines set out by ASTM 3064:
Define two paths in the x, y plane. One mainly along x the other along y.
The bar is to traverse these path in 3 different rotation. In line with the path,
vertically orthogonal to the path and horizontally orthogonal to the path. 
In total this should result in 6 distinct sets of measurements. 
(check the standard for a more detailed review)

In this experiment one orientation (vertically orthogonal) was not considerd,
as one tracker would not have only been visible to one LH. Hence, only
4 set of measurements are taken.


The bar is mounted on a robot who ensures repeatable path traversal.

Setup: 
2 LH, 2 Trackers mounted on a bar facing away from each other, the bar itself 
is mounted on a robot.

Note: The speed of the robot is varied. Furthermore, while the script waits
for the robot to move before taken measurements, no such condition is
implemented for when the robot finishes the path traversal. This has to be stopped
manually using the Keyboardinterrupt (sigint).

"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils.GS_timing import delay
from utils.general import Framework, get_file_location, save_data, check_if_moved
from utils.consts import (
    DYNAMIC_ACC_FREQUENCY, LIBSURVIVE_STABILITY_COUNTER, LIBSURVIVE_STABILITY_THRESHOLD,
    MOVING_THRESHOLD
)

if os.name == 'nt':  # if windows
    import openvr
    import utils.triad_openvr as triad_openvr
else:
    import pysurvive
    from utils.better_libsurvive_api import (
        BetterSurviveObject, get_n_survive_objects, get_simple_context, simple_start
    )


def run_dynamic_accuarcy_steamvr(
    frequency: int,
) -> np.ndarray:
    """runs the dynamic accuracy for steamvr. Collects data from both tracker
    and writes them into a matrix
    x y z w i j k x y z w i j k

    Args:
        frequency (int):
        duration (float):

    Returns:
        np.ndarray: Nx14 Matrix
    """
    v = triad_openvr.triad_openvr()
    v.print_discovered_objects()
    counter = 0
    """
    WAIT FOR tracker to move before starting measuring
    """
    print("WAITING for tracker being moved")
    inital_pose = np.array(v.devices["tracker_1"].get_pose_quaternion())
    while not check_if_moved(
        current_pose=np.array(v.devices["tracker_1"].get_pose_quaternion()),
        initial_pose=inital_pose,
        moving_threshold=MOVING_THRESHOLD
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
    print("WAITING for tracker being moved")
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
    exp_num = 5
    exp_type = "dynamic_accuracy"
    # settings:
    settings = {
        "frequency": DYNAMIC_ACC_FREQUENCY,  # Hz
        "velocity": "200 mm/s",
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
