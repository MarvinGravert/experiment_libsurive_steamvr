import sys
import time
from pathlib import Path
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
import pysurvive


class Framework(Enum):
    steamvr = "steamvr"
    libsurvive = "libsurvive"


def run_drift_libsurvive(frequency=10, duration=10*60) -> np.ndarray:
    """runs the routine to get the tracker pose from libsurive for the given duration
    in seconds with the given frequency

    Args:
        frequency (int, optional): how often measurements are taken Defaults to 10.
        duratation (int, optional): how long measurements are taken Defaults to 10.

    Returns:
        np.ndarray: x y z w i j k
    """
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


def get_file_location(exp_type, exp_num, framework: Framework) -> Path:
    exp_num = 1
    # file naming stuff
    DATA_PATH = Path("./data")
    FOLDER_PATH = Path(f"{exp_type}/{framework.value}")
    current_date = time.strftime("%Y%m%d")
    file_path = DATA_PATH/FOLDER_PATH
    file_path.mkdir(parents=True, exist_ok=True)
    file_location = file_path/Path(f"{framework.value}_{current_date}_{exp_num}.txt")
    return file_location


def save_data(file_location: Path, pose_matrix: np.ndarray, settings=dict()):
    current_date_time = time.strftime("%Y%m%d-%H%M%S")
    settings_header = ""
    for key, val in settings.items():
        settings_header += f"{key} = {val}"
    header = f"Drift {framework.value}; {current_date_time}; x y z w i j k; {settings_header}\n"
    with file_location.open("w") as file:
        file.write(header)
        np.savetxt(file, pose_matrix)


if __name__ == "__main__":
    exp_num = 1
    exp_type = "drift"
    # settings:
    settings = {
        "frequency": 10,  # Hz
        "duration": 60*10  # seconds
    }
    framework = Framework("libsurvive")
    # run program
    if framework == Framework.libsurvive:
        pose_matrix = run_drift_libsurvive(
            frequency=settings["frequency"],
            duratation=settings["duration"]
        )
    elif framework == Framework.steamvr:
        pass
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
        framework=framework
    )
    if file_location.exists():
        exit()
    save_data(
        file_location=file_location,
        pose_matrix=pose_matrix,
        settings=settings
    )
    pos_mean = np.mean(pose_matrix, 0)[:3]
    pos = pose_matrix[:, :3]
    error_pos = np.linalg.norm(pos-pos_mean, axis=1)
    plt.plot(error_pos)
    plt.show()
