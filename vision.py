"""Camera-based localisation and tower observations, with no Gazebo state input."""
from dataclasses import dataclass
import math

import cv2
import numpy as np

from geometry import (CAMERAS, INSULATOR_LENGTH, PLATE_HEIGHT, PLATE_OFFSET,
                      PLATE_SIZE, POSITIONS, marker_corners, optical_rotation)


@dataclass
class Pose:
    position: np.ndarray
    rotation: np.ndarray
    stamp: float

    @property
    def yaw(self):
        return math.atan2(self.rotation[1, 0], self.rotation[0, 0])

    def camera(self, name):
        spec = CAMERAS[name]
        return (self.position + self.rotation @ spec["translation"],
                self.rotation @ optical_rotation(spec["pitch"]))


@dataclass
class Observation:
    center: np.ndarray
    front: np.ndarray
    colors: dict
    distance: float


class ArucoLocalizer:
    def __init__(self, markers):
        self.world = {m["id"]: marker_corners(m) for m in markers}
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        if hasattr(cv2.aruco, "ArucoDetector"):
            options = cv2.aruco.DetectorParameters()
            options.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.detect = cv2.aruco.ArucoDetector(dictionary, options).detectMarkers
        else:
            options = cv2.aruco.DetectorParameters_create()
            options.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.detect = lambda image: cv2.aruco.detectMarkers(
                image, dictionary, parameters=options)

    def locate(self, image, matrix, distortion, stamp):
        corners, ids, _ = self.detect(image)
        if ids is None:
            return None
        matches = [(self.world[int(i)], c.reshape(4, 2)) for c, i in zip(corners, ids.flat)
                   if int(i) in self.world and cv2.contourArea(c) > 100]
        if not matches:
            return None
        world = np.array([p for match, _ in matches for p in match], dtype=np.float64)
        pixels = np.array([p for _, match in matches for p in match], dtype=np.float64)
        ok, vector, offset = cv2.solvePnP(
            world, pixels, matrix, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        rotation, _ = cv2.Rodrigues(vector)
        camera_rotation = rotation.T
        camera_position = -(camera_rotation @ offset).ravel()
        extrinsic = np.asarray(optical_rotation(CAMERAS["down"]["pitch"]))
        body_rotation = camera_rotation @ extrinsic.T
        position = camera_position - body_rotation @ CAMERAS["down"]["translation"]
        if not (.07 <= position[2] <= 3.5 and body_rotation[2, 2] > .86):
            return None
        reproj = cv2.projectPoints(world, vector, offset, matrix, distortion)[0]
        rms = float(np.sqrt(np.mean(np.sum((reproj[:, 0] - pixels)**2, axis=1))))
        return Pose(position, body_rotation, stamp) if rms < 3.0 else None


def plate_observations(image, matrix, distortion, pose):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (93, 125, 95), (135, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    t_camera, r_camera = pose.camera("front")
    half = PLATE_SIZE / 2
    plate = np.array([(-half, -half, 0), (half, -half, 0),
                      (half, half, 0), (-half, half, 0)], dtype=np.float64)
    observations = []
    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, .042*perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        top, bottom = np.array_split(polygon.reshape(4, 2)[np.argsort(polygon[:, 0, 1])], 2)
        tl, tr = top[np.argsort(top[:, 0])]
        bl, br = bottom[np.argsort(bottom[:, 0])]
        pixels = np.asarray([tl, tr, br, bl], dtype=np.float64)
        ok, rotations, translations, errors = cv2.solvePnPGeneric(
            plate, pixels, matrix, distortion, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            continue
        candidates = []
        for vector, translation, error in zip(rotations, translations, errors.flat):
            rotation, _ = cv2.Rodrigues(vector)
            board = t_camera + r_camera @ translation.ravel()
            normal = -r_camera @ rotation[:, 2]
            if (translation[2, 0] <= 0 or abs(board[2]-PLATE_HEIGHT) > .18
                    or abs(normal[2]) > .38 or error > 3.1):
                continue
            candidates.append((abs(normal[2])*3 + abs(board[2]-PLATE_HEIGHT)*4 + error*.15,
                               board, normal))
        if not candidates:
            continue
        _, board, normal = min(candidates, key=lambda item: item[0])
        front = normal[:2] / np.linalg.norm(normal[:2])
        # Intersect the center ray with the known height to improve range accuracy.
        center_px = np.mean(pixels, axis=0).reshape(1, 1, 2)
        point = cv2.undistortPoints(center_px, matrix, distortion)[0, 0]
        ray = r_camera @ np.array((*point, 1.))
        if ray[2] >= -.06:
            continue
        depth = (PLATE_HEIGHT-t_camera[2]) / ray[2]
        distance = float(np.linalg.norm(depth*ray))
        if distance > 5.8 or distance < .6:
            continue
        center = (t_camera + depth*ray)[:2] - PLATE_OFFSET*front
        if float(front @ (t_camera[:2]-center)) < .42:
            continue
        colors = classify(hsv, matrix, distortion, t_camera, r_camera, center, front)
        observations.append(Observation(center, front, colors, distance))
    return observations


def classify(hsv, matrix, distortion, t_camera, r_camera, center, front):
    right = np.array((-front[1], front[0]))
    positions = np.array([(*(center + side*right), z-.5*INSULATOR_LENGTH)
                          for side, z in POSITIONS.values()])
    in_camera = (positions - t_camera) @ r_camera
    pixels, _ = cv2.projectPoints(in_camera, np.zeros(3), np.zeros(3), matrix, distortion)
    colors = {}
    for number, pixel, xyz in zip(POSITIONS, pixels[:, 0], in_camera):
        u, v = np.rint(pixel).astype(int)
        if xyz[2] < .5:
            continue
        radius = max(3, min(17, round(matrix[0, 0] * .043 / xyz[2])))
        if not (radius <= u < hsv.shape[1]-radius and radius <= v < hsv.shape[0]-radius):
            continue
        patch = hsv[v-radius:v+radius+1, u-radius:u+radius+1]
        h, s, val = (patch[:, :, i] for i in range(3))
        red = ((h < 9) | (h >= 170)) & (s >= 120) & (val > 65)
        brown = (h >= 9) & (h <= 34) & (s > 82) & (val > 48)
        black = (val < 83) & (s < 150)
        votes = {"red": np.mean(red), "brown": np.mean(brown), "black": np.mean(black)}
        color = max(votes, key=votes.get)
        if votes[color] >= .20:
            colors[number] = color
    return colors
