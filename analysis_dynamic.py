
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
    Framework, load_data, get_file_location, plot_cumultive, box_plot
)
from calc_tracker_transform import TrackerTransform

import locale
locale.setlocale(locale.LC_ALL, 'de_DE.utf8')


def calc_error(data_list):
    """
    we have data of both trackers to analyse we do the following:
    1. loop over data to get the measurement for each position
     split into two
        1.1 Potentially resample
        1.2 Potentially cut data 
    2. calculate for each position the measured homogenous matrix
    5. compare to expected matrix
    """
    first_tracker_list = list()
    second_tracker_list = list()
    """ 
    1.1 Resample Or not
    """
    for pose_data in data_list:
        first_tracker_list.append(pose_data[:, :7])
        second_tracker_list.append(pose_data[:, 7:])

        # temp = resample_poly(pose_data[:, :7], 1500, len(pose_data))
        # first_tracker_list.append(temp)
        # temp = resample_poly(pose_data[:, 7:], 1500, len(pose_data))
        # second_tracker_list.append(temp)

        # temp = resample(pose_data[:, :7], 1500)
        # first_tracker_list.append(temp)
        # temp = resample(pose_data[:, 7:], 1500)
        # second_tracker_list.append(temp)
        # print(len(first_tracker_list[0]))
    """
    1.2 CUTTIGN DATA depending on a set dict (1->only first 19sec etc)
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
    each data measured point has to transformed to homogenous matrix! 
    """
    expected_hom = TrackerTransform().get_tracker_2_tracker(scaling=0.001)

    first_hom_list = list()
    second_hom_list = list()
    for first, second in zip(first_tracker_list, second_tracker_list):
        # first=>List[np.ndarray]
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
    """ 
        3. Compare to expected matrix
    """
    actual_hom_list = list()
    error_list = list()
    for first, second in zip(first_hom_list, second_hom_list):
        for f, s in zip(first, second):
            actual_hom = np.linalg.inv(s)@f
            actual_hom_list.append(actual_hom)
            temp = distance_between_hom_matrices(actual_hom, expected_hom)
            error_list.append(temp)

    pos_error_list = np.array(error_list)[:, 0]
    rot_error_list = np.array(error_list)[:, 1]
    return pos_error_list, rot_error_list


def analyze_error(error_list):

    error_percentile = calc_percentile_error(error_list)
    eval_data = eval_error_list(error_list=error_list)
    print(np.array(eval_data))
    # print(error_percentile)


def plot_error_axis(err_data):
    plt.figure()
    plt.plot(err_data)
    plt.show()


def plot_mult_line(data_list: List[float]):
    plot_list = list()
    low = 0
    for data in data_list:
        up = low+len(data)
        plot_list.append((low, up, data))
        low = up
    plt.figure(dpi=200)
    for i, (x_l, x_u, y) in enumerate(plot_list):
        if i == 0 or i == 2:
            plt.plot(np.arange(start=x_l, stop=x_u), y, c="r")
        else:
            plt.plot(np.arange(start=x_l, stop=x_u), y, c="r")
    plt.xlabel('Datenpunkt', fontsize=15)
    plt.ylabel('Fehler [mm]', fontsize=15)
    # plt.title('Messreihe 200\u2009mm/s :\nSteamVR', fontsize=15)
    plt.title('Messung 2:\nlibsurvive 0°', fontsize=15)
    # plt.legend(["90°", "0°"], loc="upper left")
    plt.yticks([0, 20, 40, 60, 80])
    plt.ylim(top=93)
    plt.show()


def get_single_data(
    exp_type: str,
    date: str,
    exp_num: int,
    framework: Framework,
    num_point: int
) -> np.ndarray:
    """
    Individual data sets 
    """
    data_list = list()
    file_loc = get_file_location(
        exp_type=exp_type,
        exp_num=exp_num,
        date=date,
        framework=framework,
        num_point=num_point
    )
    data_list.append(load_data(file_location=file_loc))
    print(f"{framework}: appended {num_point}")
    return data_list[0]


def get_all_data(
    exp_type: str,
    date: str,
    exp_num: int,
    framework: Framework,
) -> List[np.ndarray]:
    """
    All data
    """
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
            print(f"{framework}: appended {num_point}")
            num_point += 1
    except OSError:
        # end the loop if thee is no more data in the directory
        pass
    return data_list


if __name__ == "__main__":
    """
        SteamVR
    """
    exp_type = "dynamic_accuracy"
    date = "20211020"
    exp_num = 4
    framework = Framework("steamvr")
    steam_data = get_all_data(
        exp_type=exp_type,
        exp_num=exp_num,
        framework=framework,
        date=date
    )
    err_pos_steamvr, err_rot_steamvr = calc_error(data_list=steam_data)
    analyze_error(err_pos_steamvr)
    # analyze_error(err_rot_steamvr)
    """
        Libsurvive
    """
    exp_type = "dynamic_accuracy"
    date = "20211020"
    exp_num = 4
    framework = Framework("libsurvive")
    libsurvive_data = get_all_data(
        exp_type=exp_type,
        exp_num=exp_num,
        framework=framework,
        date=date
    )
    err_pos_libsurvive, err_rot_libsurvive = calc_error(data_list=libsurvive_data)

    err_pos_steamvr = np.array(err_pos_steamvr)*1000
    err_pos_libsurvive = np.array(err_pos_libsurvive)*1000

    # analyze_error(err_pos_steamvr)
    # analyze_error(err_rot_steamvr)
    # analyze_error(err_pos_libsurvive)
    # analyze_error(err_rot_libsurvive)

    # plot_cumultive(
    #     data=[err_pos_libsurvive, err_pos_steamvr]
    # )
    # box_plot(
    #     data=[err_pos_libsurvive, err_pos_steamvr]
    # )

    """
        Plot mulitple experiments same framework (speed increase)
    """
    # speed_err = list()
    # for i in range(1, 5):
    #     exp_type = "dynamic_accuracy"
    #     date = "20211020"
    #     exp_num = i
    #     framework = Framework("libsurvive")
    #     t = get_all_data(
    #         exp_type=exp_type,
    #         exp_num=exp_num,
    #         framework=framework,
    #         date=date
    #     )

    #     speed_err.append(np.array(calc_error(t))[0]*1000)  # take position and not rotation
    # plot_cumultive(speed_err)
    """ plot the error along the data points and split them up into measurements sets
    """
    exp_type = "dynamic_accuracy"
    date = "20211020"
    exp_num = 1
    framework = Framework("libsurvive")
    single_data_list = list()
    len_list = list()
    # for i in range(1, 5):
    #     test = get_single_data(
    #         exp_type=exp_type,
    #         exp_num=exp_num,
    #         framework=framework,
    #         date=date,
    #         num_point=i
    #     )
    #     single_data_list.append(test)
    #     len_list.append(len(test))
    # single experiment plot
    test = get_single_data(
        exp_type=exp_type,
        exp_num=exp_num,
        framework=framework,
        date=date,
        num_point=2
    )
    single_data_list.append(test)
    err_pos, err_rot = calc_error(data_list=single_data_list)
    # last entry is not necessary as it represents the overall lenght
    splits = np.cumsum(len_list)[:-1]
    err_pos = np.array(err_pos)*1000
    plot_mult_line(np.split(ary=err_pos, indices_or_sections=splits))
    """ plot reduced libsurvive
    """
    # exp_type = "dynamic_accuracy"
    # date = "20211020"
    # exp_num = 1
    # framework = Framework("steamvr")
    # single_data_list = list()
    # for i in [1, 3]:
    #     exp_type = "dynamic_accuracy"
    #     date = "20211020"
    #     exp_num = 1
    #     framework = Framework("libsurvive")
    #     t = get_single_data(
    #         exp_type=exp_type,
    #         exp_num=exp_num,
    #         framework=framework,
    #         date=date,
    #         num_point=i
    #     )
    #     single_data_list.append(t)

    # red_lib_pos_err, red_lib_rot_err = calc_error(single_data_list)

    # red_lib_pos_err = np.array(red_lib_pos_err)*1000
    # red_lib_rot_err = np.array(red_lib_rot_err)
    # analyze_error(red_lib_pos_err)
    # analyze_error(red_lib_rot_err)
    # # plot_cumultive(
    # #     data=[err_pos_libsurvive, err_pos_steamvr, red_lib_pos_err]
    # # )
    # box_plot(
    #     data=[err_pos_libsurvive, err_pos_steamvr, red_lib_pos_err]
    # )
