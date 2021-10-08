import matplotlib.pyplot as plt
import ctypes
from pathlib import Path
import time
import sys

import numpy as np
import pysurvive

# get data
# settings: measure on 10 Hz for 10min. Start actually taking data after 30seconds
frequency = 50  # Hz
duration = 5  # 60*number minutes=seconds
# run program
actx = pysurvive.SimpleContext(sys.argv)
time.sleep(5)

obj_dict = dict()
while actx.Running():
    updated = actx.NextUpdated()
    if updated is not None:
        if updated.Name() not in obj_dict:
            obj_dict[updated.Name()] = updated
            print(f"objects: ", updated.Name())
    if len(obj_dict.keys()) == 3:
        break
counter = 0
max_counter = duration*frequency
pose_matrix = np.zeros((int(max_counter), 7))
print("START Measuring")
time.sleep(1)
while actx.Running() and counter < max_counter:
    current_time = time.perf_counter()

    pose, _ = obj_dict[b'T20'].Pose()
    pos = np.array([i for i in pose.Pos])
    rot = np.array([i for i in pose.Rot])
    pose_matrix[counter, :] = np.hstack((pos, rot))
    try:
        time_2_sleep = 1/frequency-(time.perf_counter()-current_time)
        time.sleep(time_2_sleep)
    except ValueError:  # happends if negative sleep duration (loop took too long)
        print("hhhh")
        pass
    counter += 1
print("Done measureing")
# save data
# file naming stuff
test_run = 0
point_number = 10  # 1-10 corresponds to the point number in the robot program
counter = 9  # 0-9
DATA_PATH = Path("./data")
FOLDER_PATH = Path(f"repeatability/libsurvive/run{test_run}")
current_date_time = time.strftime("%Y%m%d-%H%M%S")
file_path = DATA_PATH/FOLDER_PATH
file_path.mkdir(parents=True, exist_ok=True)
file_location = file_path/Path(f"libsurvive_P{point_number}_counter{counter}.txt")

# x y z w i j k
# TODO: ADD settings used to the header
header = f"Drift libsurvive; {current_date_time}; x y z w i j k; Frequency = {frequency}; Duration = {duration}\n"
with file_location.open("w") as file:
    file.write(header)
    np.savetxt(file, pose_matrix)

pos_mean = np.mean(pose_matrix, 0)[:3]
pos = pose_matrix[:, :3]
error_pos = np.linalg.norm(pos-pos_mean, axis=1)
# plt.plot(error_pos)
# plt.show()
