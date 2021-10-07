from utils.linear_algebrea_helper import (
    eval_error_list,
    calc_percentile_error
)
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# dont take norm


def read_pose_matrix(file_path: Path):
    return np.loadtxt(file_path, skiprows=1)


DATA_PATH = Path("./data")
FOLDER_PATH = Path("drift/libsurvive")
file_path = DATA_PATH/FOLDER_PATH
num = 1
date = "20211007"
naming = f"libsurvive_{date}_{num}.txt"
file_path = file_path/Path(naming)

matrix = read_pose_matrix(file_path=file_path)

cut_time = 60
frequency = 10
number_cut = cut_time*frequency
pos = matrix[number_cut:, :3]
rot = matrix[number_cut:, 3:]
pos_mean = np.mean(pos, 0)
fig, ax = plt.subplots()
diff = pos-pos_mean
for di in diff.T:
    print(eval_error_list(di*1000))
    print(calc_percentile_error(di*1000))
print(diff.shape)
norm_diff = np.linalg.norm(diff, axis=1)
print(norm_diff.shape)
print(eval_error_list(norm_diff))
print(calc_percentile_error(norm_diff))
# PLOT
leg = ax.plot(diff[:, 0], label="x")
leg = ax.plot(diff[:, 1], label="y")
leg = ax.plot(diff[:, 2], label="z")
leg = ax.plot(norm_diff, label="norm")
plt.legend()

# leg.set_label(["jalkdfjl", "lksdfj", "ldkf"])
plt.show()
