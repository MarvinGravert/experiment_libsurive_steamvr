from pathlib import Path
import numpy as np
from utils.linear_algebrea_helper import (
    build_coordinate_system_via_3_points
)


def get_laser_data():
    file_location = Path("./data", "Laser", "laser_data.txt")
    temp = np.genfromtxt(file_location, delimiter=",", skip_header=1)
    return temp[:, :3]
