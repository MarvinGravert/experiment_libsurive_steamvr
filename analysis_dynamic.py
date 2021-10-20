
from pathlib import Path
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation as R, rotation
from scipy.signal import resample_poly, resample
import matplotlib.pyplot as plt
from more_itertools import chunked

from utils.linear_algebrea_helper import (
    calc_percentile_error, distance_between_hom_matrices, eval_error_list,
    transform_to_homogenous_matrix)
from utils.averageQuaternions import averageQuaternions
from utils.general import (
    Framework, load_data, get_file_location
)
from laser_trans import TransformLibsurvive, TransformSteamVR


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


def analyze_data(data_list):
    """
    we have data of both trackers to analyse we do the following:
    1. loop over data to get the measurement for each position
    2. split into two
    3. put back into list (as hom)
    4. calculate for each position the measured homogenous matrix
    5. compare to expected matrix
    6. Analyze the difference
    """
    first_tracker_list = list()
    second_tracker_list = list()
    """ 
    Split and Average
    """
    for pose_data in data_list:
        first_tracker_list.append(pose_data[:, :7])
        second_tracker_list.append(pose_data[:, 7:])
        # temp = resample_poly(pose_data[:, :7], 1500, len(pose_data))
        # first_tracker_list.append(temp)
        # temp = resample_poly(pose_data[:, 7:], 1500, len(pose_data))
        # second_tracker_list.append(temp)

    #     temp = resample(pose_data[:, :7], 2500)
    #     first_tracker_list.append(temp)
    #     temp = resample(pose_data[:, 7:], 2500)
    #     second_tracker_list.append(temp)
    # print(len(first_tracker_list[0]))
    """
    CUTTIGN DATA depending on a set dict (1->only first 19sec etc)
    """
    # cuttingStart = {
    #     1: 0,
    #     2: 0,
    #     3: 0,
    #     4: 0
    # }
    # cuttingEnd = {
    #     1: -1,
    #     2: 1700,
    #     3: 1800,
    #     4: 1700
    # }
    # first_tracker_list = [pose_data[cuttingStart[i]:cuttingEnd[i], :]
    #                       for i, pose_data in enumerate(first_tracker_list, start=1)]
    # second_tracker_list = [pose_data[cuttingStart[i]:cuttingEnd[i], :]
    #                        for i, pose_data in enumerate(second_tracker_list, start=1)]
    """ 
    Calculate homgenous matrix and compare to expected
    """
    if framework == Framework.libsurvive:
        expected_hom = TransformLibsurvive().get_tracker_2_tracker(scaling=0.001)
        expected_hom = TransformSteamVR().get_tracker_2_tracker(scaling=0.001)
    elif framework == Framework.steamvr:
        expected_hom = TransformSteamVR().get_tracker_2_tracker(scaling=0.001)

    first_hom_list = list()
    second_hom_list = list()
    for first, second in zip(first_tracker_list, second_tracker_list):
        first_hom_list.append([transform_to_homogenous_matrix(
            position=f[:3],
            quaternion=f[3:],
            scalar_first=True
        )for f in first])
        second_hom_list.append([transform_to_homogenous_matrix(
            position=s[:3],
            quaternion=s[3:],
            scalar_first=True
        )for s in second])
    actual_hom_list = list()
    error_list = list()
    for first, second in zip(first_hom_list, second_hom_list):
        for f, s in zip(first, second):
            actual_hom = np.linalg.inv(s)@f
            actual_hom_list.append(actual_hom)
            temp = distance_between_hom_matrices(actual_hom, expected_hom)
            error_list.append(temp)

    pos_error_list = np.array(error_list)[:, 0]*1000
    rot_error_list = np.array(error_list)[:, 1]
    print(eval_error_list(pos_error_list))
    print(len(pos_error_list))
    # print(eval_error_list(rot_error_list))
    # plot_cumultive([pos_error_list])
    # plt.plot(first_tracker_list[3][:, :3])
    plt.plot(pos_error_list)
    plt.show()


def plot_cumultive(data: List[List[float]]):
    # def plot_cumultive_distribution(data_points: List[float], relevant_data: float):
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
    plt.xlabel('Mittleres Abstandsquadrat [mm]', fontsize=15)
    plt.ylabel('Kumulative Häufigkeit', fontsize=15)

    # Min: {minVal:n}mm Max: {maxVal:n}mm
    plt.title('Roboter-Referenzierung RMSE:\n'+f'{acc:n}mm\u00B1{std:n}mm', fontsize=15)
    # TODO: check naming kumulativer Fehler, evtl Verteilungsfunktion? siehe Normalverteilung
    print(len(x))
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
    plt.vlines(acc, ymin=0, ymax=2, colors="r")
    ticky.append(acc)
    # plt.ticklabel_format(useLocale=True)
    # add stuff
    # plt.xticks(ticky)
    plt.ylim(ymin=0, ymax=1.05)
    plt.xlim(xmin=0)
    plt.show()


if __name__ == "__main__":
    exp_type = "dynamic_accuracy"
    date = "20211014"
    exp_num = 2
    framework = Framework("steamvr")
    data_list = list()
    """
    Individual data sets 
    """
    # file_loc = get_file_location(
    #     exp_type=exp_type,
    #     exp_num=exp_num,
    #     date=date,
    #     framework=framework,
    #     num_point=3
    # )
    # data_list.append(load_data(file_location=file_loc))
    """
    All data
    """
    try:
        num_point = 1
        while True:
            file_loc = get_file_location(
                exp_type=exp_type,
                exp_num=exp_num,
                date=date,
                framework=framework,
                num_point=num_point
            )
            data_list.append(load_data(file_location=file_loc))
            num_point += 1
    except OSError:
        # end the loop if thee is no more data in the directory
        pass
    analyze_data(data_list)
