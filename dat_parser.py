import re
from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
file_loc = Path("./MarStaticAcc.dat")
with file_loc.open("r") as file:
    t = file.read().replace('\n', '')

pattern = 'XP[0-9]*=\{[^S]*'
result = re.findall(pattern, t)
""" 
list of the point coordinates. 
Example:
XP1={X 530.457397,Y 75.6248474,Z 750.328064,A -165.776642,B 0.319188625,C 0.329238772,
We are interested in X Y Z, soo 
"""
waypoint_list = list()
for test in result:

    coordinates = test.split("X ")[1].split(",A")[0]
    # remove Y Z and space
    for i in (("Y", ""), ("Z", ""), (" ", "")):
        coordinates = coordinates.replace(*i)

    coor = [float(i) for i in coordinates.split(",")]
    waypoint_list.append(coor)

waypoint_matrix = np.array(waypoint_list)


def plot_robo_calibration_points(rob_points):
    min = np.amin(rob_points, axis=0)
    max = np.amax(rob_points, axis=0)

    fig = plt.figure(figsize=(8, 6), dpi=150)
    # fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=18, azim=-159)

    for i, row in enumerate(rob_points):
        if i < 26:
            ax.scatter(row[0], row[1], row[2], c="orange")
            # ax.text(row[0], row[1], row[2],  '%s' % (str(i+1)), size=10, zorder=1,  color='k')
        else:
            ax.scatter(row[0], row[1], row[2], c="brown")
        ax.text(row[0], row[1], row[2],  '%s' % (str(i+1)), size=10, zorder=1,  color='k')

    ax.set_xlabel('x [mm]', fontsize=10)
    ax.set_ylabel('y [mm]', fontsize=10)
    # import upsidedown
    # test = upsidedown.transform('z [mm]')
    # ax.set_zlabel(test, fontsize=10)
    # ax.set_zticks([0, 400, 500, 600, 700, 800])
    # BUILD CUBE
    center = min+(max-min)/2
    size = max-min+[20, 20, 20]
    # https://stackoverflow.com/questions/30715083/python-plotting-a-wireframe-3d-cuboid

    ox, oy, oz = center
    l, w, h = size
    # logger.error(f"{ox}, {oy}, {oz}")

    x = np.linspace(ox-l/2, ox+l/2, num=2)
    y = np.linspace(oy-w/2, oy+w/2, num=2)
    z = np.linspace(oz-h/2, oz+h/2, num=2)
    x1, z1 = np.meshgrid(x, z)
    y11 = np.ones_like(x1)*(oy-w/2)
    y12 = np.ones_like(x1)*(oy+w/2)
    x2, y2 = np.meshgrid(x, y)
    z21 = np.ones_like(x2)*(oz-h/2)
    z22 = np.ones_like(x2)*(oz+h/2)
    y3, z3 = np.meshgrid(y, z)
    x31 = np.ones_like(y3)*(ox-l/2)
    x32 = np.ones_like(y3)*(ox+l/2)

    ax = fig.gca(projection='3d')
    # outside surface
    ax.plot_wireframe(x1, y11, z1, color='b', rstride=1, cstride=1, alpha=0.6)
    # inside surface
    ax.plot_wireframe(x1, y12, z1, color='b', rstride=1, cstride=1, alpha=0.6)
    # bottom surface
    ax.plot_wireframe(x2, y2, z21, color='b', rstride=1, cstride=1, alpha=0.6)
    # upper surface
    ax.plot_wireframe(x2, y2, z22, color='b', rstride=1, cstride=1, alpha=0.6)
    # left surface
    ax.plot_wireframe(x31, y3, z3, color='b', rstride=1, cstride=1, alpha=0.6)
    # right surface
    ax.plot_wireframe(x32, y3, z3, color='b', rstride=1, cstride=1, alpha=0.6)

    plt.show()


test = waypoint_matrix[:, :3]
plot_robo_calibration_points(test)
