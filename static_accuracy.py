import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils.general import Framework, get_file_location, save_data
from GS_timing import delay
if os.name == 'nt':  # if windows
    import openvr
    import utils.triad_openvr as triad_openvr
else:
    import pysurvive


def run_static_accuracy_steamvr(
    frequency: int,
    duration: float,
) -> np.ndarray:
    """runs the static accuracy for steamvr. Collects data from both tracker
    and writes them into a matrix
    x y z w i j k x y z w i j k

    Args:
        frequency (int, optional): (max) measuring frequency 
        duration (float, optional): time during which the tracker are polled

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


def get_pose_libsurvive_obj(pose_obj) -> np.ndarray:
    pos = np.array([i for i in pose_obj.Pos])  # try np.array(x,dtype=float)
    rot = np.array([i for i in pose_obj.Rot])
    return np.hstack((pos, rot))


def run_static_accuracy_libsurvive(
    frequency: int,
    duration: float,
) -> np.ndarray:
    """runs the static accuracy for libsurvive. Collects data from both tracker
    and writes them into a matrix
    x y z w i j k x y z w i j k

    Args:
        frequency (int, optional): (max) measuring frequency 
        duration (float, optional): time during which the tracker are polled
    Returns:
        np.ndarray: nx14 matrix containing poses of both trackers
    """
    actx = pysurvive.SimpleContext(sys.argv)
    time.sleep(5)
    # collect all objects
    obj_dict = dict()
    while actx.Running():
        updated = actx.NextUpdated()
        if updated is not None:
            if updated.Name() not in obj_dict:
                obj_dict[updated.Name()] = updated
                print(f"objects: ", updated.Name())
        if len(obj_dict.keys()) == 4:
            break
    counter = 0
    max_counter = duration*frequency
    print("START Measuring")
    time.sleep(1)
    tracker_obj_1 = obj_dict[b'T20']
    tracker_obj_2 = obj_dict[b'T21']
    pose_list = list()
    while actx.Running() and counter < max_counter:
        current_time = time.perf_counter()
        pose_1, _ = tracker_obj_1.Pose()
        pose_2, _ = tracker_obj_2.Pose()
        pose_list.append([pose_1, pose_2])
        counter += 1
        try:
            time_2_sleep = 1/frequency-(time.perf_counter()-current_time)
            time.sleep(time_2_sleep)
        except ValueError:  # happends if negative sleep duration (loop took too long)
            pass
    pose_matrix = np.zeros((int(max_counter), 7))
    for j, (pose_1, pose_2) in enumerate(pose_list):
        pose_1 = get_pose_libsurvive_obj(pose_1)
        pose_2 = get_pose_libsurvive_obj(pose_2)
        pose_matrix[j, :] = np.hstack((pose_1, pose_2))
    return pose_matrix


if __name__ == "__main__":
    exp_num = 1
    exp_type = "static_accuracy"
    # settings:
    settings = {
        "frequency": 150,  # Hz
        "duration": 2  # seconds
    }
    framework = Framework("steamvr")

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
        pose_matrix = run_drift_libsurvive(
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
    if num_point > 2:
        num_point += 1  # adjusting because i deleted P3 in the .src and getting confused^^
    print(f"Run for point {num_point}")
    print("Finished")
