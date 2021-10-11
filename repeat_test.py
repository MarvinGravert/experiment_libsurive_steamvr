import os
import sys
import time
from pathlib import Path

from utils.general import Framework, get_file_location, save_data
import numpy as np

if os.name == 'nt':  # if windows
    import openvr
    import utils.triad_openvr as triad_openvr
else:
    import pysurvive


def run_libsurvive_repeatability(
    frequency: float,
    duration: float
) -> np.ndarray:

    actx = pysurvive.SimpleContext(sys.argv)
    time.sleep(5)

    obj_dict = dict()
    while actx.Running():
        updated = actx.NextUpdated()
        if updated is not None:
            if updated.Name() not in obj_dict:
                obj_dict[updated.Name()] = updated
                print(f"objects: ", updated.Name())
        if len(obj_dict.keys()) == 3:
            break
    counter = 0
    max_counter = duration*frequency
    pose_matrix = np.zeros((int(max_counter), 7))
    print("START Measuring")
    time.sleep(1)
    tracker_obj = obj_dict[b'T20']
    pose_list = list()
    while actx.Running() and counter < max_counter:
        current_time = time.perf_counter()
        pose, _ = tracker_obj.Pose()
        pose_list.append(pose)
        try:
            time_2_sleep = 1/frequency-(time.perf_counter()-current_time)
            time.sleep(time_2_sleep)
        except ValueError:  # happends if negative sleep duration (loop took too long)
            pass
        counter += 1
    for j, pose in enumerate(pose_list):
        pos = np.array([i for i in pose.Pos])
        rot = np.array([i for i in pose.Rot])
        pose_matrix[j, :] = np.hstack((pos, rot))
    return pose_matrix


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
    pose_matrix = np.zeros((int(max_counter), 7))
    print("START Measuring")
    time.sleep(1)
    pose_list = list()
    while counter < max_counter:
        current_time = time.perf_counter()
        pose = v.devices["tracker_1"].get_pose_quaternion()
        pose_list.append(pose)
        try:
            time_2_sleep = 1/frequency-(time.perf_counter()-current_time)
            time.sleep(time_2_sleep)
        except ValueError:  # happends if negative sleep duration (loop took too long)
            pass
        counter += 1
    for j, pose in enumerate(pose_list):
        pose_matrix[j, :] = np.array(pose)
    return pose_matrix


if __name__ == "__main__":
    point_number = 1  # 1-10
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
