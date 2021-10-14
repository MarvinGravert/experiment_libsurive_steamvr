from utils.linear_algebrea_helper import (
    eval_error_list,
    calc_percentile_error
)
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from utils.general import Framework, get_file_location, load_data
# dont take norm

if __name__ == "__main__":
    exp_num = 1
    exp_type = "drift"
    num_point = 1
    date = "20211007"
    framework: Framework = Framework("libsurvive")

    file_location = get_file_location(
        exp_type=exp_type,
        exp_num=exp_num,
        framework=framework,
        num_point=num_point,
        date=date
    )
    matrix = load_data(file_location,
                       exp_type=exp_type,
                       framework=framework)

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
