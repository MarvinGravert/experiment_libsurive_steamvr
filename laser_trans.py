"""Script to extract the relative position between the two trackers mounted on the 
bar.

The tracker holder has holes for laser targets to be put into. Using a laser
tracker these were used to construct two KOS. So as to get the transformation
between them. 

The transformation between the tracker holders in combination with the 
position of the trackers on the tracker holders is used to calculate the 
relative transformation between the two trackers which is used as a reference
for the static and dynamic accuarcy experiments
"""
from pathlib import Path

import numpy as np

from utils.linear_algebrea_helper import (
    build_coordinate_system_via_3_points
)


def get_laser_data():
    file_location = Path("./data", "Laser", "laser_data.txt")
    temp = np.genfromtxt(file_location, delimiter=",", skip_header=1)
    return temp[:, :3]


class TrackerTransform():
    """class describing the transformation between the two trackers
    The coordinate system of the tracker is the same in libsurive and 
    steamvr hence one can be used for each
    """
    T_tracker_to_lt1 = np.array([[1, 0, 0, 25],
                                [0, 1, 0, 85],
                                [0, 0, 1, -9],
                                [0, 0, 0, 1]])

    T_lt2_to_tracker = np.array([[-1, 0, 0, 25],
                                [0, 1, 0, -85],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])

    def get_laser_hom(self) -> np.ndarray:
        """calculates the matrix between the tracker holders based on the
        laser tracker measurements. 

        the data file contains two sets of measurements each consisting of
        8 points
        1 coordinate system: origin->x->y->origin
        2 coordinate system: origin->x->y->origin

        Returns:
            np.ndarray: 4x4 homogenous matrix
        """
        laser_data = get_laser_data()
        first_kos = laser_data[8:11, :]
        second_kos = laser_data[4:7, :]
        first_T = build_coordinate_system_via_3_points(*first_kos)
        sec_T = build_coordinate_system_via_3_points(*second_kos)
        first_2_second = np.linalg.inv(sec_T)@first_T
        return first_2_second

    def get_tracker_2_tracker(self, scaling=1) -> np.ndarray:
        """get the transformation between the two trackers. the positional part
        is in mm but one may use scaling to transform to meters

        Args:
            scaled (int, optional): scales the translational partn. Defaults to 1.

        Returns:
            np.ndarray: [description]
        """
        first_2_second = self.get_laser_hom()
        temp = self.T_lt2_to_tracker@first_2_second@self.T_tracker_to_lt1
        temp[:3, 3] = temp[:3, 3]*scaling
        return temp


if __name__ == "__main__":
    laser_data = get_laser_data()
    first_kos = laser_data[8:11, :]
    second_kos = laser_data[4:7, :]
    first_T = build_coordinate_system_via_3_points(*first_kos)
    sec_T = build_coordinate_system_via_3_points(*second_kos)
    diff_T = np.linalg.inv(sec_T)@first_T
    # print(diff_T[:3, 3])
    # print(np.linalg.norm(diff_T[:3, 3]))
    # print(distance_between_hom_matrices(first_T, sec_T))
    print(diff_T)
    """ 
    1. get the KOS from the laser points
    2. Calculate the transformation matrix between the KOS
    3. Calculate the distance from
    """
    t = TrackerTransform()
    test = t.T_lt2_to_tracker@diff_T@t.T_tracker_to_lt1
    print(test)
