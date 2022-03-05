from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.stats import ttest_ind

from utils.general import Framework, get_file_location, load_data, plot_cumultive, box_plot
from utils.averageQuaternions import averageQuaternions
from utils.linear_algebrea_helper import (
    calc_percentile_error, distance_between_rotation_matrices,
    eval_error_list, rotational_distance_quaternion,
)


def get_repeatability_data(
    exp_type: str,
    date: str,
    framework: Framework,
    start_point: int,
    end_point: int
) -> List[List[np.ndarray]]:
    """for the defined range of measuring points, return the data associated with these points


    point_list= for reach position (point) considered for analysis
    there is data for 10 repetition from different directions
    pointlist (x points)
        data (<10 points)
            nx7

    Args:
        exp_type (str): [description]
        date (str): [description]
        framework (Framework): [description]
        start_point (int): [description]
        end_point (int): [description]
    Returns:
        List[List[np.ndarray]]: across all positions, all associated measurement matrices
    """
    point_list = list()
    for point in range(start_point, end_point+1):
        data_list = list()
        num_point = 1
        while True:
            try:
                file_loc = get_file_location(
                    exp_type=exp_type,
                    exp_num=point,
                    date=date,
                    framework=framework,
                    num_point=num_point
                )
                data = load_data(file_location=file_loc)
                data_list.append(data)
            except OSError:
                break
            num_point += 1
        point_list.append(data_list)
    return point_list


def get_precision_distance(
    data: List[List[np.ndarray]]
) -> List[float]:
    """calculate the translational and rotational error associated with the 
    repeatability data gathered in the experiment

    calculate for each designated measurement position 
        1. the mean of the measurements conducted at the position
        1.1. If needed, data can be cut
        2. The mean of the means (center mean across all measurements for the position)
        3. the distance of each mean to the center mean 
        4. add the normed distance to a list
    Args:
        data (List[List[np.ndarray]]): measurment data across all points associated with each point

    Returns:
        Tuple[List[float],List[float]]: translational and rotational error 
    """
    number_cut = 0
    norm_pos_list = list()
    norm_rot_list = list()
    for point in data:  # 10 position
        mean_pos_list = list()
        mean_rot_list = list()
        # 1
        for measurement in point:  # <=10 measurements per position
            cut_data = measurement[number_cut:, :]
            mean_pos_list.append(np.mean(cut_data[:, :3], 0))
            mean_rot_list.append(averageQuaternions(cut_data[:, 3:]))
        # 2
        center_pos_mean = np.mean(mean_pos_list, 0)
        center_rot_mean = averageQuaternions(np.array(mean_rot_list))
        # 3
        distance_pos_vectors = np.array(mean_pos_list)-center_pos_mean
        # 4
        norm_pos_list.extend(np.linalg.norm(distance_pos_vectors, axis=1))
        norm_rot_list.extend([rotational_distance_quaternion(quat_a=i, quat_b=center_rot_mean,
                             scalar_first=True) for i in mean_rot_list])

    return norm_pos_list, norm_rot_list


def run_analysis(
    error_data: List[float]
):
    eval_data = eval_error_list(error_list=error_data)
    error_percentile = calc_percentile_error(error_data)
    print(np.array(eval_data))
    # print(error_percentile)


def plot_distances():
    """plot for each point the mean distance norm 
    steamvr vs libsurive vs (libsurive sans imu)
    """
    pass


if __name__ == "__main__":
    exp_num = 1  # 1-10 are the points and 11-20 the next
    exp_type = "repeatability"
    date = "20211020"
    framework = Framework("steamvr")
    data_list = get_repeatability_data(
        exp_type=exp_type,
        date=date,
        framework=framework,
        start_point=1,
        end_point=10
    )
    pos_err_steamvr, rot_err_steamvr = get_precision_distance(data=data_list)
    print("Steamvr")
    run_analysis(pos_err_steamvr)
    run_analysis(rot_err_steamvr)
    exp_num = 1  # 1-10 are the points and 11-20 the next
    exp_type = "repeatability"
    date = "20211016"
    framework = Framework("libsurvive")
    data_list = get_repeatability_data(
        exp_type=exp_type,
        date=date,
        framework=framework,
        start_point=1,
        end_point=10
    )
    pos_err_libsurvive, rot_err_libsurvive = get_precision_distance(data=data_list)

    print("libsurvive")
    run_analysis(pos_err_libsurvive)
    run_analysis(rot_err_libsurvive)

    print("ttest")
    print(ttest_ind(
        a=pos_err_libsurvive,
        b=pos_err_steamvr,
        equal_var=False,
        alternative="less"
    ))
    pos_err_libsurvive = np.array(pos_err_libsurvive)*1000
    pos_err_steamvr = np.array(pos_err_steamvr)*1000
    # plot_cumultive(
    #     data=[pos_err_libsurvive, pos_err_steamvr]
    # )

    # box_plot(pos_err_libsurvive, pos_err_steamvr)
    mean_pos_list = list()
    for point in data_list:  # 10 position
        # 1
        for measurement in point:  # <=10 measurements per position
            cut_data = measurement[0:, :]
            mean_pos_list.append(np.mean(cut_data[:, :3], 0))
        # 2

    t = np.array(mean_pos_list)
    print(np.ptp(t, axis=0))
    print(t.shape)
