from pathlib import Path
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation as R, rotation

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


if __name__ == "__main__":
    exp_type = "static_accuracy"
    date = "20211014"
    exp_num = 1
    framework = Framework("steamvr")
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
    """
    we have data of both trackers to analyse we do the following:
    1. loop over data to get the measurement for each position
    2. split into two and average each 
    3. put back into list
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
        first_tracker_list.append(average_pose(pose_data[:, :7]))
        second_tracker_list.append(average_pose(pose_data[:, 7:]))
    """ 
    Calculate homgenous matrix and compare to expected
    """
    if framework == Framework.libsurvive:
        expected_hom = TransformLibsurvive().get_tracker_2_tracker(scaling=0.001)
    elif framework == Framework.steamvr:
        expected_hom = TransformSteamVR().get_tracker_2_tracker(scaling=0.001)

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
    actual_hom_list = list()
    error_list = list()
    for first, second in zip(first_hom_list, second_hom_list):
        actual_hom = np.linalg.inv(second)@first
        actual_hom_list.append(actual_hom)
        temp = distance_between_hom_matrices(actual_hom, expected_hom)
        error_list.append(temp)

    pos_error_list = np.array(error_list)[:, 0]
    rot_error_list = np.array(error_list)[:, 1]
    print(eval_error_list(rot_error_list))
    t = eval_error_list(pos_error_list)
    print(t)
    print(pos_error_list)
