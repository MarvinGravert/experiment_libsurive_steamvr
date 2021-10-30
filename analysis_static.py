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
    Framework, load_data, get_file_location, plot_cumultive, average_pose, box_plot
)
from calc_tracker_transform import TrackerTransform

import locale
locale.setlocale(locale.LC_ALL, 'de_DE.utf8')


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
    plt.figure(dpi=200)
    # x = np.arange(len(err_libsurvive))
    # plt.scatter(x, err_libsurvive, c="b")
    # plt.scatter(x, err_steam, c="r")
    n_steam = np.arange(len(err_steam))
    n_lib = np.arange(len(err_libsurvive))
    plt.plot(err_libsurvive)
    plt.scatter(n_lib, err_libsurvive,)
    plt.plot(err_steam)
    plt.scatter(n_steam, err_steam, )
    ax = plt.gca()
    ax.set_xlabel('Posenummer', fontsize=10)
    ax.set_ylabel('Fehler [mm]', fontsize=10)
    plt.ticklabel_format(useLocale=True)
    plt.legend(["libsurvive", "SteamVR"])
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
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    """only color specific lines!
    """
    # for k, (first, second) in enumerate(zip(first_tracker_list, second_tracker_list)):
    #     if k in (0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 32, 33):
    #         ax.plot(
    #             [first[0], second[0]],
    #             [first[1], second[1]],
    #             [first[2], second[2]],
    #             color="b"
    #         )
    #         ax.scatter(first[0], first[1], first[2], c="b")
    #         ax.scatter(second[0], second[1], second[2], c="b")
    #     else:
    #         ax.plot(
    #             [first[0], second[0]],
    #             [first[1], second[1]],
    #             [first[2], second[2]],
    #             color="r"
    #         )
    #         ax.scatter(first[0], first[1], first[2], c="r")
    #         ax.scatter(second[0], second[1], second[2], c="r")
    """for all color gradient
    """
    colormap = cm.gist_rainbow
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

    # x, y, z = np.array(first_tracker_list)[:, :3].T
    # p1 = ax.scatter(x, y, z, c=normed, cmap=plt.cm.cool)
    # x, y, z = np.array(second_tracker_list)[:, :3].T
    # p2 = ax.scatter(x, y, z, c=normed, cmap=plt.cm.cool)
    x1, y1, z1 = np.array(first_tracker_list)[:, :3].T
    x2, y2, z2 = np.array(second_tracker_list)[:, :3].T
    x = np.hstack((x1, x2))
    y = np.hstack((y1, y2))
    z = np.hstack((z1, z2))
    normed2 = np.hstack((normed, normed))
    print(x.shape, normed2.shape)
    p2 = ax.scatter(x, y, z, c=normed2, cmap=colormap)
    p1 = ax.scatter(x, y, z, c=colormap(normed2), cmap=colormap)
    cbar = fig.colorbar(p2, ax=ax,)
    cbar.mappable.set_clim(np.min(error_data), np.max(error_data))

    """general setup
    """
    ax.set_xlabel('x [m]', fontsize=10)
    ax.set_ylabel('y [m]', fontsize=10)
    # import upsidedown
    # test = upsidedown.transform('z [mm]')
    ax.set_zlabel('z [m]', fontsize=10)
    ax.ticklabel_format(useLocale=True)
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

    exp_type = "static_accuracy"
    date = "20211020"
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

    err_pos_steamvr = np.array(err_pos_steamvr)*1000
    err_pos_libsurvive = np.array(err_pos_libsurvive)*1000
    analyze_error(err_pos_steamvr)
    analyze_error(err_rot_steamvr)

    analyze_error(err_pos_libsurvive)
    analyze_error(err_rot_libsurvive)

    # plot_cumultive([err_pos_libsurvive, err_pos_steamvr])
    # plot_error_axis(err_pos_steamvr, err_pos_libsurvive)
    # print(err_pos_steamvr)
    plot_bar(data_list_steamvr, error_data=err_pos_libsurvive)
    # box_plot([err_pos_libsurvive, err_pos_steamvr])
