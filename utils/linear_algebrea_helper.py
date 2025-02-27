from typing import Tuple, Union, List
from matplotlib.pyplot import sca
from scipy.spatial.transform import Rotation as R
import numpy as np


def combine_to_homogeneous_matrix(
        rotation_matrix: np.ndarray, translation_vector: np.ndarray) -> np.ndarray:
    """combine rotation matrix and translation vector together to create a 4x4
    homgeneous matrix

    Args:
        rotation_matrix (np.ndarray): 3x3 matrix
        translation_vector (np.ndarray): 3x1 or (3,) vector

    Returns:
        np.ndarray: 4x4 homogeneous matrix
    """
    temp = np.hstack((rotation_matrix, translation_vector.reshape((-1, 1))))
    return np.vstack((temp, [0, 0, 0, 1]))


def separate_from_homogeneous_matrix(homogenous_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """separate the rotation matrix and translation vector from the homogenous matrix

    Args:
        homogenous_matrix (np.ndarray): 4x4 homogenous matrix

    Returns:
        Tuple[np.ndarray, np.ndarray]: 3x3 rotation matrix, (3,1) rotation vector
    """
    return homogenous_matrix[:3, :3], homogenous_matrix[:3, 3]


def extract_position_and_quaternion_from_homogeneous_matrix(
        homogenous_matrix: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray]:
    """extract position and quaternion representation from a 4x4 homogeneous matrix

    Args:
        homogenous_matrix (np.ndarray): 4x4 homogeneous

    Returns:
        Tuple[np.ndarray, np.ndarray]: (3,)position, (4,) quaternion i j k w
    """
    r, t = separate_from_homogeneous_matrix(homogenous_matrix)
    quat = R.from_matrix(r).as_quat()
    return t, quat


def transform_to_homogenous_matrix(
    position: Union[np.ndarray, List[float]],
    quaternion: Union[np.ndarray, List[float]],
    scalar_first: bool = False
) -> np.ndarray:
    """given a position and a quaternion calculates the homogeneous representation

    Args:
        position (Union[np.ndarray, List[float]]): x y z position
        quaternion (Union[np.ndarray, List[float]]): quaternion i j k w
        scalar_first (bool): use w i j k instead
    Returns:
        np.ndarray: 4x4 homogeneous matrix
    """
    if scalar_first:
        w, i, j, k = quaternion
        quaternion = [i, j, k, w]
    r = R.from_quat(quaternion)
    p = np.array(position).reshape((-1, 1))
    temp = np.hstack((r.as_matrix(), p))
    return np.vstack((temp, [0, 0, 0, 1]))


def calc_reprojection_error(
    point_set_a: np.ndarray,
    point_set_b: np.ndarray,
    hom_matrix: np.ndarray
) -> Tuple[float]:
    """calculates the reprojection error of a transformation between two point
    sets and returns root mean squared, absolute mean, max deviation and standard deviation

    a={a_i}, b={b_i}
    e_i=||b_i-T*a_i|| # T:4x4 hom matrix
    RMSE: e = sqrt(sum(e_i**2)/num_points)
    MAE: e = sum(e_i)/num_points #mittlerer absoluter Fehler

    Args:
        point_set_a (np.ndarray): nx3 or nx4 matrix of points
        point_set_b (np.ndarray): nx3 or nx4 mastrix of points
        hom_matrix (np.ndarray): 4x4 homogenous matrix mapping from a->b

    Returns:
        Tuple[float]: root mean square error, mean average error, max, std
    """
    num_points = point_set_a.shape[0]
    # First check and possible augment
    if point_set_a.shape[1] == 3:
        ones = np.ones((num_points, 1))
        point_set_a = np.hstack((point_set_a, ones))
        point_set_b = np.hstack((point_set_b, ones))

    # lets do it weirdly just to be sure numpy is doing it correclty
    error = np.empty_like(point_set_a)
    for i in range(num_points):
        error[i, :] = point_set_b[i, :]-hom_matrix@point_set_a[i, :]

    # cut augmented column
    error = error[:, :3]
    sum_squared_error = 0
    sum_error = 0
    err_list = list()
    for i in range(num_points):
        err_mag = np.linalg.norm(error[i, :3])
        err_list.append(err_mag)
        sum_error += err_mag
        sum_squared_error += err_mag**2

    std = np.std(err_list)
    rmse = np.sqrt(sum_squared_error/num_points)
    mae = sum_error/num_points
    return rmse, mae, max(np.abs(err_list)), std


def eval_error_list(error_list) -> Tuple[float]:
    """calculates rmse mae average_error, std and max of the error list handed in.
    Error list is assumed to be the magnitude of the difference between predicted-actual

    rmse=root mean squared error
    mae=mean absolute error
    std= standard deviation of the non squared error
    Args:
        error_list ([type]): 1dim array of error values

    Returns:
        Tuple[float]: rmse, mae, avg_error std and max
    """
    num_points = len(error_list)
    sum_squared_error = 0
    err_list = list()
    for err in error_list:
        err_list.append(err)
        sum_squared_error += err**2
    std = np.std(err_list)

    rmse = np.sqrt(sum_squared_error/num_points)
    mae = np.mean(np.abs(err_list))
    avg_error = np.mean(err_list)
    return rmse, mae, avg_error, std, max(np.abs(err_list))


def calc_percentile_error(
        error_list: List[float],
        percentiles: List[float] = [50, 95, 99.7]
) -> Tuple[float]:
    """calcutates the percentile error for the given data

    Args:
        error_list (List[float]): [description]

    Returns:
        Tuple[float]: [description]
    """
    return [np.percentile(error_list, per) for per in percentiles]


def single_reprojection_error(point_a, point_b, hom_matrix) -> float:
    """calculates the reprojection error for given hom matrix between
    two points correspodnance

    Args:
        point_set_a (np.ndarray): 1x3 or 1x4 point
        point_set_b (np.ndarray): 1x3 or 1x4 mastrix of points
        hom_matrix (np.ndarray): 4x4 homogenous matrix mapping from a->b

    Returns:
        float: error
    """
    if point_a.shape[1] == 3:
        ones = np.ones((1, 1))
        point_a = np.hstack((point_a, ones))
        point_b = np.hstack((point_b, ones))

    error = point_b-hom_matrix@point_a

    return np.linalg.norm(error)


def reprojection_error_axis_depending(point_set_a: np.ndarray,
                                      point_set_b: np.ndarray,
                                      hom_matrix: np.ndarray):
    """calculate the reprojection error in the axis of the target system

    Returns the rmse and mae axis depening (aka 2 times 3dim vector)

    Args:
        point_set_a (np.ndarray): nx3 or nx4 mastrix of points
        point_set_b (np.ndarray):  nx3 or nx4 mastrix of points
        hom_matrix (np.ndarray):  4x4 homogenous matrix mapping from a->b

    Returns:
        List[np.ndarray, np.ndarray]: rmse and mae as 1x3 vectors
    """
    num_points = point_set_a.shape[0]
    # First check and possible augment
    if point_set_a.shape[1] == 3:
        ones = np.ones((num_points, 1))
        point_set_a = np.hstack((point_set_a, ones))
        point_set_b = np.hstack((point_set_b, ones))

    # lets do it weirdly just to be sure numpy is doing it correclty
    error = np.empty_like(point_set_a)
    for i in range(num_points):
        error[i, :] = point_set_b[i, :]-hom_matrix@point_set_a[i, :]

    # cut augmented column
    error = error[:, :3]
    abs_error = np.abs(error)
    mean_abs_error = np.mean(abs_error, 0)
    squared_error = error**2
    mean_squared_error = np.mean(squared_error, 0)
    rmse = np.sqrt(mean_squared_error)
    return rmse, mean_abs_error


def get_angle_from_rot_matrix(rot_matrix) -> float:
    # https://en.wikipedia.org/wiki/Rotation_matrix#Determining_the_angle
    temp = (np.trace(rot_matrix)-1)/2
    return np.rad2deg(np.arccos(temp))


def build_coordinate_system_via_3_points(origin, x_axis_point, y_axis_point) -> np.ndarray:
    """utilising 3 non colinear points build a coordinate system with its origin
    and axis lying at the coordinates handed over


    Args:
        origin (np.ndarray): 3D point
        x_axis_point (np.ndarray): 3D point along the new axis
        y_axis_point (np.ndarray): 3D point along the new yaxis

    Returns:
        np.ndarray: Returns the transformation from new to old coordinate system
    """
    x_axis = x_axis_point-origin
    x_axis = x_axis/np.linalg.norm(x_axis)
    y_axis = y_axis_point-origin
    y_axis = y_axis/np.linalg.norm(y_axis)
    # build z axis
    z_axis = np.cross(x_axis, y_axis)
    y_axis = np.cross(z_axis, x_axis)  # make sure 90° between x and y
    rot = np.column_stack((x_axis, y_axis, z_axis))
    return combine_to_homogeneous_matrix(rotation_matrix=rot, translation_vector=origin)


def distance_between_hom_matrices(
    actual_hom: np.ndarray,
    ideal_hom: np.ndarray
) -> Tuple[float]:
    """calculates the relative distance between two homgenous matrices by
    comparing the translational and rotational distance they apply to points.
    Returns the normed translational and angular distances
    Args:
        actual_hom (np.ndarray): 4x4 homogenous matrix
        ideal_hom (np.ndarray): 4x4 homogenous matrix

    Returns:
        Tuple[float]: distance, angle (in deg)
    """
    r1, t1 = separate_from_homogeneous_matrix(actual_hom)
    r2, t2 = separate_from_homogeneous_matrix(ideal_hom)
    alpha1 = get_angle_from_rot_matrix(r1)
    alpha2 = get_angle_from_rot_matrix(r2)
    dist1 = np.linalg.norm(t1)
    dist2 = np.linalg.norm(t2)
    # TODO: theoretically we are not allowd to take absolute
    return np.abs(dist1-dist2), np.abs(alpha1-alpha2)


def distance_between_rotation_matrices(
    rot_matrix_b: np.ndarray,
    rot_matrix_c: np.ndarray
) -> float:
    """calculates the distance between two rotations matrices which are related 
    by a common system. Returns distance as the angle necessary to rotate into one another
    first: B->A
    second: C->A
    results in B->C which is  

    Args:
        rot_matrix_b (np.ndarray): [description]
        rot_matrix_c (np.ndarray): [description]

    Returns:
        float: [description]
    """
    diff_rot = np.linalg.inv(rot_matrix_b)@rot_matrix_c
    return get_angle_from_rot_matrix(diff_rot)


def quaternion_angle_rotation(
    quat: Union[np.ndarray, List[float]],
    scalar_first=False
) -> float:
    """calculatest the angle of rotation of a quaternion in DEGREE
    standard format i j k w 
    if scalar_true is set=> format is w i j k
    Args:
        quat (Union[np.ndarray,List[float]]): quat 
        scalar_first (bool): changes expected format to w i j k 
    Returns:
        float: angle of rotation in DEGREE
    """
    if scalar_first:
        w, i, j, k = quat
    else:
        i, j, k, w = quat
    temp = np.linalg.norm([i, j, k])
    return np.rad2deg(2*np.arcsin(temp))


def rotational_difference_quaternion(
    quat_a: Union[np.ndarray, List[float]],
    quat_b: Union[np.ndarray, List[float]],
    scalar_first=False
) -> float:
    """calculates the rotational difference between two quaternion in DEGREE!

    Args:
        quat_a (np.ndarray):  i j k w 
        quat_b (np.ndarray):  i j k w 
        scalar_first (bool, optional): changes format to w i j k. Defaults to False.

    Returns:
        float: angle in DEGREE
    """
    angle_a = quaternion_angle_rotation(quat=quat_a, scalar_first=scalar_first)
    angle_b = quaternion_angle_rotation(quat=quat_b, scalar_first=scalar_first)
    return np.abs(angle_a-angle_b)


def rotational_distance_quaternion(
    quat_a: Union[np.ndarray, List[float]],
    quat_b: Union[np.ndarray, List[float]],
    scalar_first=False
) -> float:
    """calculates the rotational distance between two quaternion in DEGREE!

    assumes that both quaternion describe the rotation to a common base coordinate
    aka quat_a=>A->C
    quat_b=>B-C

    Calculates the relative transformation A->B and evaluates the distance
    in terms of translation and rotation

    Args:
        quat_a (np.ndarray):  i j k w 
        quat_b (np.ndarray):  i j k w 
        scalar_first (bool, optional): changes format to w i j k. Defaults to False.

    Returns:
        float: angle in DEGREE
    """
    if scalar_first:
        w, i, j, k = quat_a
        rot_a = R.from_quat([i, j, k, w])
        w, i, j, k = quat_b
        rot_b = R.from_quat([i, j, k, w])
    else:
        rot_a = R.from_quat(quat_a)
        rot_b = R.from_quat(quat_b)
    r = rot_b.inv()*rot_a
    return get_angle_from_rot_matrix(r.as_matrix())


def quaternion_to_rot_matrix(
    quat: Union[np.ndarray, List[float]],
    scalar_first=False
) -> np.ndarray:
    """transforms a quaternion into the equivalent 3x3 rotation matrix

    Args:
        quat (Union[np.ndarray, List[float]]): i j k w quaternion
        scalar_first (bool, optional): sets the expected quat to w i j k. Defaults to False.

    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    if scalar_first:
        w, i, j, k = quat
    else:
        i, j, k, w = quat

    rot = R.from_quat([i, j, k, w])
    return rot.as_matrix()
