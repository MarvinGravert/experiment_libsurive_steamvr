import os
import sys
import time
from pathlib import Path
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt

if os.name == 'nt':
    import openvr
    import utils.triad_openvr as triad_openvr
else:
    import pysurvive


class Framework(Enum):
    steamvr = "steamvr"
    libsurvive = "libsurvive"


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


def get_file_location(exp_type, exp_num, framework: Framework, num_point: int = None) -> Path:
    # file naming stuff
    DATA_PATH = Path("./data")
    FOLDER_PATH = Path(f"{exp_type}/{framework.value}")
    current_date = time.strftime("%Y%m%d")
    file_path = DATA_PATH/FOLDER_PATH/Path(f"{current_date}_{exp_num}")
    file_path.mkdir(parents=True, exist_ok=True)
    file_location = file_path/Path(f"{framework.value}_{num_point}.txt")
    return file_location


def save_data(file_location: Path, pose_matrix: np.ndarray, exp_type: str, settings=dict(), ):
    current_date_time = time.strftime("%Y%m%d-%H%M%S")
    settings_header = ""
    for key, val in settings.items():
        settings_header += f" {key} = {val}"
    header = f"{exp_type} {framework.value}; {current_date_time}; x y z w i j k;{settings_header}\n"
    with file_location.open("w") as file:
        file.write(header)
        np.savetxt(file, pose_matrix)


if __name__ == "__main__":
    exp_num = 1
    exp_type = "static_accuracy"
    # settings:
    settings = {
        "frequency": 100,  # Hz
        "duration": 1  # seconds
    }
    # TODO: Build look ahead!!
    num_point = 3
    framework = Framework("steamvr")
    # run program
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
    file_location = get_file_location(
        exp_type=exp_type,
        exp_num=exp_num,
        framework=framework,
        num_point=num_point
    )
    if file_location.exists():
        print("file exists")
    else:
        save_data(
            file_location=file_location,
            pose_matrix=pose_matrix,
            exp_type=exp_type,
            settings=settings
        )
    # pos_mean = np.mean(pose_matrix, 0)[:3]
    # pos = pose_matrix[:, :3]
    # error_pos = np.linalg.norm(pos-pos_mean, axis=1)
    # plt.plot(error_pos)
    # plt.show()
