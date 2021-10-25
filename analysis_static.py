from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R, rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import colors
from matplotlib import cm

from utils.linear_algebrea_helper import (
    calc_percentile_error, distance_between_hom_matrices, eval_error_list,
    transform_to_homogenous_matrix)
from utils.general import (
    Framework, load_data, get_file_location, plot_cumultive, average_pose
)
from calc_tracker_transform import TrackerTransform


def calc_error(data_list: List[np.ndarray]) -> Tuple[float, float]:
    """
    we have data of both trackers to analyse we do the following:
    1. loop over data to get the measurement for each position
       split into two trackers and average each
    2. calculate for each position the transformation between the trackers
    3. compare to expected matrix
    Args:
        data_list (np.ndarray): List of the N measurements for the N points
    """
    first_tracker_list = list()
    second_tracker_list = list()
    """ 
        1. Split and Average
    """
    for pose_data in data_list:
        first_tracker_list.append(average_pose(pose_data[:, :7]))
        second_tracker_list.append(average_pose(pose_data[:, 7:]))
    """ 
        2. Calculate homgenous matrices 
    """

    first_hom_list = list()
    second_hom_list = list()
    for first, second in zip(first_tracker_list, second_tracker_list):
        first_hom_list.append(transform_to_homogenous_matrix(
            position=first[:3],
            quaternion=first[3:],
            scalar_first=True
        ))
        second_hom_list.append(transform_to_homogenous_matrix(
            position=second[:3],
            quaternion=second[3:],
            scalar_first=True
        ))
    """ 
        3. Compare to expected matrix
    """
    actual_hom_list = list()
    expected_hom = TrackerTransform().get_tracker_2_tracker(scaling=0.001)
    error_list = list()
    for first, second in zip(first_hom_list, second_hom_list):
        actual_hom = np.linalg.inv(second)@first
        actual_hom_list.append(actual_hom)
        temp = distance_between_hom_matrices(actual_hom, expected_hom)
        error_list.append(temp)

    pos_error_list = np.array(error_list)[:, 0]
    rot_error_list = np.array(error_list)[:, 1]
    return pos_error_list, rot_error_list


def analyze_error(error_list):
    print(eval_error_list(error_list=error_list))


def plot_error_axis(err_steam, err_libsurvive):
    plt.figure()
    plt.plot(err_steam)
    plt.plot(err_libsurvive)
    plt.show()


def plot_bar(pose_data, error_data=list()):
    first_tracker_list = list()
    second_tracker_list = list()
    """ 
        1. Split and Average
    """
    for pose_data in data_list:
        first_tracker_list.append(average_pose(pose_data[:, :7]))
        second_tracker_list.append(average_pose(pose_data[:, 7:]))
    # error_data = np.round((np.array(error_data)/np.max(error_data))*1000, 0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    """only color specific lines!
    """
    # for k, (first, second) in enumerate(zip(first_tracker_list, second_tracker_list)):
    #     if k in (0, 1, 2, 3, 4, 5, 6, 7, 17, 32, 33):
    #         ax.plot(
    #             [first[0], second[0]],
    #             [first[1], second[1]],
    #             [first[2], second[2]],
    #             color="b"
    #         )
    #     else:
    #         ax.plot(
    #             [first[0], second[0]],
    #             [first[1], second[1]],
    #             [first[2], second[2]],
    #             color="r"
    #         )
    # norm = colors.Normalize()
    # norm.autoscale(error_data)
    colormap = cm.cool
    # colormap = plt.cm.get_cmap('RdYlGn')
    # colormap = cm.RdYlGn
    normed = np.array(error_data)/np.max(error_data)

    for first, second, error in zip(first_tracker_list, second_tracker_list, normed):
        test = ax.plot(
            [first[0], second[0]],
            [first[1], second[1]],
            [first[2], second[2]],
            color=colormap(error)
        )

    ax.set_xlabel('x [m]', fontsize=10)
    ax.set_ylabel('y [m]', fontsize=10)
    # import upsidedown
    # test = upsidedown.transform('z [mm]')
    ax.set_zlabel('z [m]', fontsize=10)
    x, y, z = np.array(first_tracker_list)[:, :3].T
    p = ax.scatter(x, y, z, c=normed, cmap=plt.cm.cool)
    x, y, z = np.array(second_tracker_list)[:, :3].T
    p = ax.scatter(x, y, z, c=normed, cmap=plt.cm.cool)
    cbar = fig.colorbar(p, ax=ax)
    cbar.mappable.set_clim(np.min(error_data)*1000, np.max(error_data)*1000)
    plt.show()


if __name__ == "__main__":
    exp_type = "static_accuracy"
    date = "20211014"
    exp_num = 1
    framework = Framework("steamvr")
    data_list_steamvr = list()
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
            data_list_steamvr.append(load_data(file_location=file_loc))
            num_point += 1
    except OSError:
        # end the loop if thee is no more data in the directory
        pass

    err_pos_steamvr, err_rot_steamvr = calc_error(data_list=data_list_steamvr)
    analyze_error(err_pos_steamvr)
    analyze_error(err_rot_steamvr)
    exp_type = "static_accuracy"
    date = "20211015"
    exp_num = 1
    framework = Framework("libsurvive")
    data_list = list()
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
    err_pos_libsurvive, err_rot_libsurvive = calc_error(data_list=data_list)

    analyze_error(err_pos_libsurvive)
    analyze_error(err_rot_libsurvive)

    # plot_cumultive([err_pos_libsurvive, err_pos_steamvr])
    # plot_error_axis(err_pos_steamvr, err_pos_libsurvive)
    # print(err_pos_steamvr)
    plot_bar(data_list_steamvr, error_data=err_pos_libsurvive)
