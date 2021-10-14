import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils.general import Framework, get_file_location, save_data
from utils.linear_algebrea_helper import (
    distance_between_hom_matrices,
    transform_to_homogenous_matrix
)
if os.name == 'nt':  # if windows
    import openvr
    import utils.triad_openvr as triad_openvr
else:
    import pysurvive


def check_if_moved(
    current_pose: np.ndarray,
    initial_pose: np.ndarray,
    moving_threshold: float = 0.1
) -> bool:
    """check if the object has moved from its initial pose 

    Args:
        pose (np.ndarray): current pose
        initial_pose (np.ndarray): initial pose
        moving_threshold (float, optional): distance in m considered moved. Defaults to 0.1.

    Returns:
        bool: [description]
    """
    pos, rot = current_pose[:3], current_pose[3:]
    ini_pos, ini_rot = initial_pose[:3], initial_pose[3:]
    current_T = transform_to_homogenous_matrix(
        position=pos,
        quaternion=rot,
        scalar_first=True
    )
    init_T = transform_to_homogenous_matrix(
        position=ini_pos,
        quaternion=ini_rot,
        scalar_first=True
    )
    diff_pos, _ = distance_between_hom_matrices(current_T, init_T)
    if diff_pos > moving_threshold:
        return True
    else:
        return False


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
    pose_list = list()
    try:
        while True:
            current_time = time.perf_counter()
            pose_tracker_1 = v.devices["tracker_1"].get_pose_quaternion()
            pose_tracker_2 = v.devices["tracker_2"].get_pose_quaternion()
            counter += 1  # counter before adding because it may otherwise cause a matrix limit expection later on in case of interrupt
            pose_list.append([pose_tracker_1, pose_tracker_2])
            try:
                time_2_sleep = 1/frequency-(time.perf_counter()-current_time)
                time.sleep(time_2_sleep)
            except ValueError:  # happends if negative sleep duration (loop took too long)
                pass
    except KeyboardInterrupt:
        endtime = time.perf_counter()
        print(
            f"Stopped measuring. After {counter} meausrements in roughly {endtime-starttime} seconds")
    pose_matrix = np.zeros((int(counter), 14))
    for j, pose in enumerate(pose_list):
        pose_matrix[j, :] = np.array(pose[0]+pose[1])
    return pose_matrix


def get_pose_libsurvive_obj(pose_obj) -> np.ndarray:
    pos = np.array([i for i in pose_obj.Pos])  # try np.array(x,dtype=float)
    rot = np.array([i for i in pose_obj.Rot])
    return np.hstack((pos, rot))


def run_dynamic_accuarcy_libsurvive(
    frequency: int,
    duration: float,
) -> np.ndarray:
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
    print("START Measuring")
    tracker_obj_1 = obj_dict[b'T20']
    tracker_obj_2 = obj_dict[b'T21']
    initial_pose = get_pose_libsurvive_obj(tracker_obj_1.Pose())
    while check_if_moved(
        initial_pose=initial_pose,
        current_pose=get_pose_libsurvive_obj(pose_obj=tracker_obj_1.Pose()),
        moving_threshold=0.1
    ):
        time.sleep(0.1)
    pose_list = list()
    while actx.Running():
        current_time = time.perf_counter()
        pose_1, _ = tracker_obj_1.Pose()
        pose_2, _ = tracker_obj_2.Pose()
        counter += 1
        pose_list.append([pose_1, pose_2])
        try:
            time_2_sleep = 1/frequency-(time.perf_counter()-current_time)
            time.sleep(time_2_sleep)
        except ValueError:  # happends if negative sleep duration (loop took too long)
            pass
    pose_matrix = np.zeros((int(counter), 7))
    for j, (pose_1, pose_2) in enumerate(pose_list):
        pose_1 = get_pose_libsurvive_obj(pose_1)
        pose_2 = get_pose_libsurvive_obj(pose_2)
        pose_matrix[j, :] = np.hstack((pose_1, pose_2))
    return pose_matrix


if __name__ == "__main__":
    exp_num = 1
    exp_type = "dynamic_accuracy"
    # settings:
    settings = {
        "frequency": 150,  # Hz
        "velocity": "200 mm/s"
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
