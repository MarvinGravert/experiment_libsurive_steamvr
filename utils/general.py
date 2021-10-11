import time
from enum import Enum
from pathlib import Path

import numpy as np


class Framework(Enum):
    steamvr = "steamvr"
    libsurvive = "libsurvive"


def get_file_location(
    exp_type: str,
    exp_num: int,
    framework: Framework,
    num_point: int
) -> Path:
    # file naming stuff
    DATA_PATH = Path("./data")
    FOLDER_PATH = Path(f"{exp_type}/{framework.value}")
    current_date = time.strftime("%Y%m%d")
    file_path = DATA_PATH/FOLDER_PATH/Path(f"{current_date}_{exp_num}")
    file_path.mkdir(parents=True, exist_ok=True)
    file_location = file_path/Path(f"{num_point}.txt")
    return file_location


def save_data(
        file_location: Path,
        pose_matrix: np.ndarray,
        exp_type: str,
        framework: Framework,
        settings=dict(),
):
    current_date_time = time.strftime("%Y%m%d-%H%M%S")
    settings_header = ""
    for key, val in settings.items():
        settings_header += f" {key} = {val}"
    header = f"{exp_type} {framework.value}; {current_date_time}; x y z w i j k; {settings_header}\n"
    with file_location.open("w") as file:
        file.write(header)
        np.savetxt(file, pose_matrix)


def load_data(
        file_location: Path,
        exp_type: str,
        framework: Framework,
) -> np.ndarray:
    return np.genfromtxt(file_location, delimiter=" ", skip_header=1)
