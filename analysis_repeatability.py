from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from utils.general import Framework, get_file_location, load_data
from utils.averageQuaternions import averageQuaternions
from utils.linear_algebrea_helper import (
    calc_percentile_error, distance_between_rotation_matrices,
    eval_error_list
)


def get_point_list(
    exp_type: str,
    date: str,
    framework: Framework,
    start_point: int,
    end_point: int
) -> List[List[np.ndarray]]:
    """ get a list of all data contained for start till end point

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
    """
    point_list = list()
    for point in range(start_point, end_point+1):
        start = 1
        stop = 10  # TODO: CHange this to work with any number of points pmeasured for positions
        data_list = list()
        for num_point in range(start, stop+1):
            file_loc = get_file_location(
                exp_type=exp_type,
                exp_num=point,
                date=date,
                framework=framework,
                num_point=num_point
            )
            data = load_data(file_location=file_loc)
            data_list.append(data)
        point_list.append(data_list)
    return point_list


def dist_quaternion(q1, q2, scalarFirst=False):
    if scalarFirst:
        w, i, j, k = q1  # probably better done with .pop() oh well
        q1 = np.array([i, j, k, w])
        w, i, j, k = q2
        q2 = np.array([i, j, k, w])
        rot1 = R.from_quat(q1)
        rot2 = R.from_quat(q2)
        return distance_between_rotation_matrices(rot1.as_matrix(), rot2.as_matrix())
    else:
        pass


def run_percentile_analysis(
    exp_type: str,
    date: str,
    framework: Framework
):
    """
    Settings for Analysis
    """
    start_point = 1
    end_point = 1
    """
    Start the data extraction

    point_list= for reach position (point) considered for analysis
    there is data for 10 repetition from different directions
    pointlist (x points)
        data (10 points)
            nx7
    """
    point_list = get_point_list(
        exp_type=exp_type,
        date=date,
        framework=framework,
        start_point=start_point,
        end_point=end_point
    )
    """ 
    Analysis start
    calculate for each position 
        1. the mean position of the each 10 measurements
        1.1. Cut away the first N measurements!
        2. The mean of the mean position (center mean)
        3. the distance vector of each mean to the center mean 
        4. the norm of the distance vectors (<=10 normed distance vectors per point)
    Add the normed distances to a list 
    Use it to calculate RMSE, MAE, Average, percentile etc.
    """
    number_cut = 0
    norm_pos_list = list()
    norm_rot_list = list()
    for point in point_list:  # 10 position
        mean_pos_list = list()
        mean_rot_list = list()
        for measurement in point:  # <=10 measurements per position
            cut_data = measurement[number_cut:, :]
            mean_pos = np.mean(cut_data[:, :3], 0)
            mean_rot = averageQuaternions(cut_data[:, 3:])
            mean_pos_list.append(mean_pos)
            mean_rot_list.append(mean_rot)
        center_pos_mean = np.mean(mean_pos_list, 0)
        center_rot_mean = averageQuaternions(np.array(mean_rot_list))
        distance_pos_vectors = np.array(mean_pos_list)-center_pos_mean
        norm_pos_list.extend(np.linalg.norm(distance_pos_vectors, axis=1))
        norm_rot_list.extend([dist_quaternion(q1=i, q2=center_rot_mean,
                             scalarFirst=True) for i in mean_rot_list])

    percentile_pos = calc_percentile_error(norm_pos_list)
    percentile_rot = calc_percentile_error(norm_rot_list)
    eval_pos = eval_error_list(norm_pos_list)
    eval_rot = eval_error_list(norm_rot_list)

    print(np.array(eval_pos)*1000)
    print(np.array(percentile_pos)*1000)


def plot_distances():
    """plot for each point the mean distance norm 
    steamvr vs libsurive vs (libsurive sans imu)
    """
    pass


if __name__ == "__main__":
    exp_num = 1  # 1-10 are the points and 11-20 the next
    exp_type = "repeatability"
    date = "20211006"
    framework = Framework("libsurvive")
    run_percentile_analysis(
        exp_type=exp_type,
        date=date,
        framework=framework
    )
