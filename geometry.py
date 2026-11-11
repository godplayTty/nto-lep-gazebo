"""Dimensions and camera extrinsics shared by the world and ROS nodes."""

import math

WORLD = "lep"
FRAME = "aruco_map"
BODY = "base_link"
PLATE_SIZE = 0.20
PLATE_HEIGHT = 0.60
PLATE_OFFSET = 0.237
POSITIONS = {1: (0.60, 1.20), 2: (-0.60, 1.20),
             3: (-0.60, 0.90), 4: (0.60, 0.90)}
INSULATOR_LENGTH = 0.15
STATES = {"black": "Исправен", "brown": "Износ", "red": "Дефект"}
CAMERAS = {
    "down": {"translation": (0.02, 0., -0.058), "pitch": math.pi / 2,
             "width": 640, "height": 480, "fov": 1.98},
    "front": {"translation": (0.13, 0., -0.025), "pitch": 0.46,
              "width": 960, "height": 720, "fov": 1.42},
}


def marker_map():
    return [
        {"id": i, "x": x, "y": y, "z": 0.008, "size": 0.32}
        for i, (x, y) in enumerate((x, y) for x in range(-2, 16)
                                    for y in range(-6, 7))
    ]


def marker_corners(marker):
    x, y, z, half = marker["x"], marker["y"], marker["z"], marker["size"] / 2
    return [(x-half, y+half, z), (x+half, y+half, z),
            (x+half, y-half, z), (x-half, y-half, z)]


def optical_rotation(pitch):
    """Rotation from ROS optical coordinates (right, down, forward) to body."""
    c, s = math.cos(pitch), math.sin(pitch)
    return ((0., -s, c), (-1., 0., 0.), (0., -c, -s))
