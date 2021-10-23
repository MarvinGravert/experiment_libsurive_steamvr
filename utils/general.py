"""Collection of helper functions and class
"""
import time
from enum import Enum
from typing import List
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from more_itertools import chunked
from utils.averageQuaternions import averageQuaternions


class Framework(Enum):
    steamvr = "steamvr"
    libsurvive = "libsurvive"


def get_file_location(
    exp_type: str,
    exp_num: int,
    framework: Framework,
    num_point: int,
    date: str = None
) -> Path:
    """returns the location of the data file for the given spefication. 
    If the folder doesnt exist, creates it.
    Used to save or load data. To load data, a date will need to be specified.

    Args:
        exp_type (str): drift, repeatability, static, dynamic
        exp_num (int): 1+
        framework (Framework): libsurive or steamvr
        num_point (int): data point within the experiment
        date (str, optional): date of the file. Used to load old data. Defaults to None.

    Returns:
        Path: Path to file
    """
    # file naming stuff
    DATA_PATH = Path("./data")
    FOLDER_PATH = Path(f"{exp_type}/{framework.value}")
    if date:
        current_date = date
    else:
        current_date = time.strftime("%Y%m%d")
    file_path = DATA_PATH/FOLDER_PATH/Path(f"{current_date}_{exp_num}")
    file_path.mkdir(parents=True, exist_ok=True)
    file_location = file_path/Path(f"{num_point}.txt")
    return file_location


def save_data(
        file_location: Path,
        pose_matrix: np.ndarray,
        exp_type: str,
        framework: Framework,
        settings=dict(),
):
    """Save the data inside the pose_matrix into the file location.
    Experiment type, framework and setting are used to create a header
    for the file

    Args:
        file_location (Path): location where to save data
        pose_matrix (np.ndarray): data matrix to be saved
        exp_type (str): drift, repeatability, static, dynamic
        framework (Framework): libsurvive or steamvr
        settings ([str:Any], optional): settingsfile. Defaults to dict().
    """
    current_date_time = time.strftime("%Y%m%d-%H%M%S")
    settings_header = ""
    for key, val in settings.items():
        settings_header += f" {key} = {val}"
    header = f"{exp_type} {framework.value}; {current_date_time}; x y z w i j k; {settings_header}\n"
    with file_location.open("w") as file:
        file.write(header)
        np.savetxt(file, pose_matrix)


def load_data(
        file_location: Path,
) -> np.ndarray:
    """load the data on the given file location
    Skips the header

    Args:
        file_location (Path): file location

    Returns:
        np.ndarray: matrix: Nx7 or Nx14
    """
    return np.genfromtxt(file_location, delimiter=" ", skip_header=1)


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
        bool: True if moved
    """
    pos, rot = current_pose[:3], current_pose[3:]
    ini_pos, ini_rot = initial_pose[:3], initial_pose[3:]
    diff_pos = np.linalg.norm(pos-ini_pos)
    if diff_pos > moving_threshold:
        return True
    else:
        return False


def plot_cumultive(data: List[List[float]]):
    """Create a cumulitive plot. Each dataset inside the data list results in an 
    individual line in the graph

    Args:
        data (List[List[float]]): List of floating points representing the error
    """
    n_list = list()
    x_list = list()
    y_list = list()
    plot_list = list()
    for total in data:
        n = len(total)
        x = np.sort(total)
        y = np.arange(n)/n
        n_list.append(n)
        x_list.append(x)
        y_list.append(y)
        plot_list.append((x, y))

    acc = round(np.mean(total), 2)
    std = round(np.std(total), 2)
    minVal = round(min(total), 2)
    maxVal = round(max(total), 2)
    # plotting
    plt.figure(dpi=200)
    # popt, pcov = curve_fit(func, x, y)
    plt.xlabel('Fehler [mm]', fontsize=15)
    plt.ylabel('Kumulative Häufigkeit', fontsize=15)

    # Min: {minVal:n}mm Max: {maxVal:n}mm
    # plt.title('Statische Genauigkeit: :\n'+f'{acc:n}mm\u00B1{std:n}mm', fontsize=15)
    plt.title('Wiederholbarkeit', fontsize=15)
    for plotty in plot_list:
        plt.scatter(x=plotty[0], y=plotty[1], marker='o')
    # plt.scatter(relevant_data, y=highlighted_y)
    # plt.plot(x, func(x, *popt), 'r-', label="Fitted Curve")
    plt.grid()
    # ticks
    ticky = list()
    for ti in chunked(x, 13):
        ticky.append(round(np.mean(ti), 0))
    # ticky = [20, 50, 80]
    # plt.xticks(np.linspace(start=min(x), stop=max(x), num=20, dtype=int))
    # accuracy line
    for acc in data:
        plt.vlines(np.mean(acc), ymin=0, ymax=2, colors="r")
    ticky.append(acc)
    # plt.ticklabel_format(useLocale=True)
    # add stuff
    # plt.xticks(ticky)
    plt.ylim(ymin=0, ymax=1.05)
    plt.xlim(xmin=0)
    plt.show()


def average_pose(pose_matrix: np.ndarray) -> np.ndarray:
    """average a pose consisting of translational and quaternion of the strucuture
    x y z w i j k

    applies standard averaging for the first 3 and quaternion averaging for the quatnernion

    Args:
        pose_matrix (np.ndarray): Nx7 matrix

    Returns:
        np.ndarray: 1x7 of averaged data
    """
    pos = pose_matrix[:, :3]
    rot = pose_matrix[:, 3:]
    pos_mean = np.mean(pos, 0)
    rot_mean = averageQuaternions(rot)

    return np.concatenate((pos_mean, rot_mean))
