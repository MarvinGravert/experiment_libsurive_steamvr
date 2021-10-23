"""Script to analyse and plot the data from the drift experiment

We have collected a time sample. If we take the average from the beginning
and utilize this as a "inital pose" we can check the deviation from this
initial pose at the end.
Furthermore, we can plot the change in pose over time
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

from utils.general import Framework, get_file_location, load_data, average_pose
from utils.linear_algebrea_helper import (
    eval_error_list,
    calc_percentile_error
)
from utils.averageQuaternions import averageQuaternions


def analyze_data(data: np.ndarray):
    """Calculate the initial mean and subtract from the dataset and retun 
    said data.

    Args:
        data (np.ndarray): Nx7 pose matrix

    Returns:
        [np.ndarray]: Nx7 pose matrix after subtracting the initial mean
    """
    frequency = 10  # Hz
    # time considered for initial mean
    initial_measurements = frequency*60  # Hz*seconds
    final_measurements = frequency*60  # Hz seconds
    initial_poses, other_poses, final_poses = np.split(
        ary=data,
        indices_or_sections=[initial_measurements, final_measurements],
        axis=0
    )
    initial_pose_mean = average_pose(initial_poses)
    initial_pos_mean = initial_pose_mean[:3]
    initial_rot_mean = initial_pose_mean[3:]

    initial_pos, final_pos = initial_poses[:, :3], final_poses[:, :3]
    initial_rot, final_rot = initial_poses[:, 3:], final_poses[:, 3:]

    centered_initial_pos = initial_pos-initial_pos_mean
    centered_final_pos = final_pos-initial_pos_mean
    centered_data_pos = data[:, :3]-initial_pos_mean

    pos_distance = np.linalg.norm(np.mean(centered_initial_pos, 0)-np.mean(centered_final_pos, 0))
    print(f"Distance: {pos_distance}")
    initial_norm = np.linalg.norm(centered_initial_pos, axis=1)
    final_norm = np.linalg.norm(centered_final_pos, axis=1)
    statis, p_value = ttest_ind(
        a=initial_norm,
        b=final_norm,
        equal_var=False,
        # alternative="greater"
    )
    print(f"For hypothesis that initial and final are the same: p_value={p_value}")
    return centered_data_pos


def plot_data(diff):
    fig, ax = plt.subplots()
    # PLOT
    norm_diff = np.linalg.norm(diff, axis=1)
    leg = ax.plot(diff[:, 0], label="x")
    leg = ax.plot(diff[:, 1], label="y")
    leg = ax.plot(diff[:, 2], label="z")
    leg = ax.plot(norm_diff, label="norm")
    plt.legend()

    # leg.set_label(["jalkdfjl", "lksdfj", "ldkf"])
    # plt.show()


if __name__ == "__main__":
    exp_num = 1
    exp_type = "drift"
    num_point = 3
    date = "20211015"
    framework: Framework = Framework("steamvr")

    file_location = get_file_location(
        exp_type=exp_type,
        exp_num=exp_num,
        framework=framework,
        num_point=num_point,
        date=date
    )
    matrix = load_data(file_location)
    steamvr_diff = analyze_data(matrix[100:, :])

    exp_num = 1
    exp_type = "drift"
    num_point = 1
    date = "20211016"
    framework: Framework = Framework("libsurvive")

    file_location = get_file_location(
        exp_type=exp_type,
        exp_num=exp_num,
        framework=framework,
        num_point=num_point,
        date=date
    )
    matrix = load_data(file_location)
    libsurvive_diff = analyze_data(matrix[100:, :])

    plot_data(steamvr_diff[:, :3])
    plot_data(libsurvive_diff[:, :3])
    plt.show()
