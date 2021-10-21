import ctypes
from typing import List, Dict
from dataclasses import dataclass, field

import pysurvive
import pysurvive.pysurvive_generated as pygen
import numpy as np

from utils.linear_algebrea_helper import (
    transform_to_homogenous_matrix
)

device_lookup_table = {
    b"LHR-2F3B022B": "red",  # with red usb cable
    b"LHR-506121A7": "black",  # with black usb cable
    b"LHB-2DD6B28C": "LH0",  # on robot control box
    b"LHB-8F3CA9A0": "LH1"

}
LP_c_char = ctypes.POINTER(ctypes.c_char)
LP_LP_c_char = ctypes.POINTER(LP_c_char)


@dataclass
class BetterSurviveObject():
    ptr: ctypes.POINTER
    name: str = field(init=False)
    id: str = field(init=False)

    def __post_init__(self):
        self.name = pygen.survive_simple_object_name(self.ptr)
        self.id = pygen.survive_simple_serial_number(self.ptr)

    def get_pose_quaternion(self) -> np.ndarray:
        pose = pygen.SurvivePose()
        pygen.survive_simple_object_get_latest_pose(self.ptr, pose)
        pos = np.array(pose.Pos, dtype=float)
        rot = np.array(pose.Rot, dtype=float)
        return np.hstack((pos, rot))

    def get_homogenous_matrix(self) -> np.ndarray:
        pose = self.get_pose_quaternion()
        pos = pose[:3]
        rot = pose[3:]
        return transform_to_homogenous_matrix(
            position=pos,
            quaternion=rot,
            scalar_first=True
        )


def get_simple_context(args) -> pygen.SurviveSimpleContext:
    argc = len(args)
    argv = (LP_c_char * (argc + 1))()
    for i, arg in enumerate(args):
        enc_arg = arg.encode('utf-8')
        argv[i] = ctypes.create_string_buffer(enc_arg)
    actx = pygen.survive_simple_init(argc, argv)
    return actx


def simple_start(actx: pygen.SurviveSimpleContext):
    pygen.survive_simple_start_thread(actx)


def get_n_survive_objects(
    actx: pygen.SurviveSimpleContext,
    num: int


) -> Dict[str, BetterSurviveObject]:
    survive_objects = dict()

    while len(survive_objects) < num:
        simple_obj = pygen.survive_simple_get_next_updated(actx)
        if simple_obj:
            serial_number = pygen.survive_simple_serial_number(simple_obj)
            if not survive_objects.get(serial_number, False):
                survive_objects[serial_number] = simple_obj
                print(
                    f"{pygen.survive_simple_object_name(simple_obj)}: {pygen.survive_simple_serial_number(simple_obj)}")
    better_survive = dict()
    print(survive_objects)
    for serial, ptr in survive_objects.items():
        key = device_lookup_table[serial]
        better_survive[key] = BetterSurviveObject(ptr)
    return better_survive


def get_pose_a_to_b(obj_a: BetterSurviveObject, obj_b: BetterSurviveObject):
    hom_a = obj_a.get_homogenous_matrix()
    hom_b = obj_b.get_homogenous_matrix()
    return np.linalg.inv(hom_b)@hom_a
