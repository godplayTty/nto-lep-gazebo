#!/usr/bin/env python3
"""ROS 2 mission controller for the Gazebo inspection quadcopter."""
from collections import Counter, deque
import json
import math
from pathlib import Path
import time

from geometry_msgs.msg import TransformStamped, Twist
import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from geometry import BODY, CAMERAS, FRAME, STATES, optical_rotation
from vision import ArucoLocalizer, classify, plate_observations

ALTITUDE = 1.86
HOME = np.zeros(2)


def time_stamp(message):
    return message.sec + message.nanosec / 1e9


def quaternion(rotation):
    axis = cv2.Rodrigues(np.asarray(rotation))[0].ravel()
    angle = np.linalg.norm(axis)
    if angle < 1e-10:
        return 0., 0., 0., 1.
    return (*map(float, axis * (math.sin(angle / 2) / angle)), math.cos(angle / 2))


def transform(parent, child, translation, rotation, stamp):
    msg = TransformStamped()
    msg.header.frame_id, msg.child_frame_id, msg.header.stamp = parent, child, stamp
    vector = msg.transform.translation
    vector.x, vector.y, vector.z = map(float, translation)
    rot = msg.transform.rotation
    rot.x, rot.y, rot.z, rot.w = quaternion(rotation)
    return msg


class TowerTrack:
    def __init__(self, index, first):
        self.name = f"T{index:02d}"
        self.samples = deque(maxlen=25)
        self.votes = {i: deque(maxlen=23) for i in range(1, 5)}
        self.add(first)

    @property
    def center(self):
        return np.median([s.center for s in self.samples], axis=0)

    @property
    def front(self):
        normal = np.mean([s.front for s in self.samples], axis=0)
        return normal / np.linalg.norm(normal)

    def add(self, observation):
        self.samples.append(observation)
        if observation.distance <= 2.8:
            self.add_colors(observation.colors)

    def add_colors(self, colors):
        for number, color in colors.items():
            self.votes[number].append(color)

    def rows(self):
        x, y = map(float, self.center)
        rows = []
        for number, colors in self.votes.items():
            if len(colors) >= 3:
                color, count = Counter(colors).most_common(1)[0]
                if count >= 3:
                    rows.append({"tower_id": self.name, "x": x, "y": y,
                                 "number": number, "color": color, "state": STATES[color]})
        return rows


class Inspection(Node):
    def __init__(self):
        super().__init__("lep_inspection")
        self.declare_parameter("marker_map", "generated/aruco_map.json")
        marks = json.loads(Path(self.get_parameter("marker_map").value).read_text())
        self.localizer = ArucoLocalizer(marks["markers"])
        self.static_tf = StaticTransformBroadcaster(self)
        self.dynamic_tf = TransformBroadcaster(self)
        stamp = self.get_clock().now().to_msg()
        self.static_tf.sendTransform([
            transform(BODY, f"{name}_optical", spec["translation"],
                      optical_rotation(spec["pitch"]), stamp)
            for name, spec in CAMERAS.items()
        ])
        retained = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.results = self.create_publisher(String, "/insulators", retained)
        self.status = self.create_publisher(String, "/inspection/status", retained)
        self.velocity = self.create_publisher(Twist, "/inspection_quad/cmd_vel", 10)
        self.enable = self.create_publisher(Bool, "/inspection_quad/enable", 10)
        self.create_service(Trigger, "/inspection/start", self.start)
        self.create_service(Trigger, "/inspection/stop", self.stop)
        self.create_service(Trigger, "/inspection/kill", self.kill)
        self.intrinsics = {}
        self.last_images = {}
        for name in CAMERAS:
            self.create_subscription(CameraInfo, f"/camera/{name}/camera_info",
                                     lambda msg, n=name: self.camera_info(n, msg),
                                     qos_profile_sensor_data)
            self.create_subscription(Image, f"/camera/{name}/image",
                                     lambda msg, n=name: self.image(n, msg),
                                     qos_profile_sensor_data)
        self.poses = deque(maxlen=30)
        self.last_fix = float("-inf")
        self.tracks = []
        self.active = None
        self.search = []
        self.search_index = 0
        self.arrived = None
        self.phase = "idle"
        self.reason = ""
        self.phase_started = self.clock_seconds()
        self.mission_started = None
        self.land_target = HOME
        self.create_timer(.10, self.step)
        self.create_timer(.50, self.publish)
        self.publish()

    def clock_seconds(self):
        return self.get_clock().now().nanoseconds / 1e9

    def phase_to(self, phase, reason=""):
        self.phase, self.reason = phase, reason
        self.phase_started = self.clock_seconds()
        self.arrived = None
        self.get_logger().info(f"{phase} {reason}".strip())
        self.publish()

    def publish(self):
        rows = [row for track in self.tracks for row in track.rows()]
        self.results.publish(String(data=json.dumps(
            {"frame_id": FRAME, "insulators": rows}, ensure_ascii=False)))
        self.status.publish(String(data=json.dumps(
            {"phase": self.phase, "reason": self.reason, "found": len(rows)})))

    def start(self, request, response):
        if self.phase not in ("idle", "complete"):
            response.message = "Миссия уже запущена или требуется перезапуск симулятора."
            return response
        if any(time.monotonic() - self.last_images.get(cam, 0) > 3 or cam not in self.intrinsics
               for cam in CAMERAS) or not (
                   self.clock_seconds() > 0 and self.velocity.get_subscription_count()
                   and self.enable.get_subscription_count()):
            response.message = "Ожидаю оба видеопотока, camera_info и мост Gazebo."
            return response
        self.tracks.clear()
        self.poses.clear()
        self.last_fix = float("-inf")
        self.active = None
        self.search = [(HOME, 0.), (HOME, -.36), (HOME, .36)]
        self.search_index = 0
        self.mission_started = self.clock_seconds()
        self.enable.publish(Bool(data=True))
        self.phase_to("takeoff")
        response.success, response.message = True, "Миссия запущена."
        return response

    def stop(self, request, response):
        if self.phase not in ("idle", "complete", "killed", "failed", "stopped", "landing"):
            pose = self.fresh_pose()
            self.land_target = pose.position[:2].copy() if pose else None
            self.phase_to("landing", "operator_stop")
            response.success = True
        response.message = "Посадка." if response.success else "Активной миссии нет."
        return response

    def kill(self, request, response):
        self.velocity.publish(Twist())
        self.enable.publish(Bool(data=False))
        self.phase_to("killed", "operator_kill")
        response.success, response.message = True, "Роторы отключены."
        return response

    def camera_info(self, name, message):
        matrix = np.asarray(message.k).reshape(3, 3)
        if matrix[0, 0] > 0 and matrix[1, 1] > 0:
            self.intrinsics[name] = matrix, np.asarray(message.d, dtype=float)

    def image(self, name, message):
        self.last_images[name] = time.monotonic()
        if name not in self.intrinsics:
            return
        if name == "front" and (self.phase not in ("search", "inspect", "overflight", "return")
                                or not self.poses):
            return
        matrix, distortion = self.intrinsics[name]
        stamp = time_stamp(message.header.stamp)
        if message.encoding not in ("rgb8", "bgr8") or message.step < message.width*3:
            return
        packed = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
        image = packed[:, :message.width*3].reshape(message.height, message.width, 3)
        if message.encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if name == "down":
            pose = self.localizer.locate(image, matrix, distortion, stamp)
            if pose is not None:
                self.poses.append(pose)
                self.last_fix = time.monotonic()
                self.dynamic_tf.sendTransform(
                    transform(FRAME, BODY, pose.position, pose.rotation, message.header.stamp))
            return

        pose = min(self.poses, key=lambda p: abs(p.stamp-stamp))
        if abs(pose.stamp-stamp) > .18:
            return
        t_camera, r_camera = pose.camera("front")
        for track in self.tracks:
            offset = t_camera[:2]-track.center
            if np.linalg.norm(offset) <= 2.8 and float(offset @ track.front) < -.85:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                track.add_colors(classify(hsv, matrix, distortion, t_camera, r_camera,
                                          track.center, track.front))
        if self.phase in ("overflight", "return"):
            return
        observations = plate_observations(image, matrix, distortion, pose)
        if self.phase == "search":
            for obs in sorted(observations, key=lambda obs: obs.distance):
                if obs.distance > 4.0 or any(np.linalg.norm(t.center-obs.center) < 1.25
                                              for t in self.tracks):
                    continue
                if self.tracks and float((-self.tracks[-1].front) @
                                         (obs.center-self.tracks[-1].center)) < 1.4:
                    continue
                self.active = TowerTrack(len(self.tracks)+1, obs)
                self.tracks.append(self.active)
                self.phase_to("inspect")
                break
        else:
            for obs in sorted(observations, key=lambda obs: np.linalg.norm(
                    self.active.center-obs.center)):
                if np.linalg.norm(self.active.center-obs.center) < .65:
                    self.active.add(obs)
                    break

    def fresh_pose(self):
        if self.poses and time.monotonic()-self.last_fix < 1.6:
            return self.poses[-1]
        return None

    def send(self, x=0., y=0., z=0., yaw=0.):
        command = Twist()
        command.linear.x, command.linear.y, command.linear.z = map(float, (x, y, z))
        command.angular.z = float(yaw)
        self.velocity.publish(command)

    def go(self, pose, xy, altitude, yaw, speed=.85):
        error = xy-pose.position[:2]
        dz = altitude-pose.position[2]
        turn = math.atan2(math.sin(yaw-pose.yaw), math.cos(yaw-pose.yaw))
        world = np.clip(np.linalg.norm(error)*1.0, 0, speed) * error / max(
            np.linalg.norm(error), .0001)
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        self.send(c*world[0]+s*world[1], -s*world[0]+c*world[1],
                  np.clip(1.25*dz, -.32, .48), np.clip(1.65*turn, -.85, .85))
        return np.linalg.norm(error) < .18 and abs(dz) < .16 and abs(turn) < .15

    def held(self, arrived, seconds):
        if not arrived:
            self.arrived = None
            return False
        now = self.clock_seconds()
        if self.arrived is None:
            self.arrived = now
        return now-self.arrived >= seconds

    def next_search(self):
        last = self.tracks[-1]
        travel = -last.front
        if len(self.tracks) >= 2:
            along = last.center-self.tracks[0].center
            travel = along / np.linalg.norm(along)
        yaw = math.atan2(travel[1], travel[0])
        self.search = [(last.center + travel*offset, yaw) for offset in
                       (-.45, .5, 1.4, 2.35, 3.3, 4.25, 5.1, 5.95, 6.6)]
        self.search = [(xy, yaw) for xy, yaw in self.search
                       if -1.5 <= xy[0] <= 14.5 and abs(xy[1]) < 5.5]
        self.search_index = 0
        self.phase_to("search")

    def step(self):
        if self.phase in ("idle", "complete", "killed", "failed", "stopped"):
            return
        elapsed = self.clock_seconds()-self.phase_started
        pose = self.fresh_pose()
        if self.phase == "takeoff" and pose is None:
            if elapsed < 8:
                self.send(z=.42)
            else:
                self.land_target = None
                self.phase_to("landing", "no_aruco_fix")
            return
        if self.phase != "landing" and (
                self.clock_seconds()-self.mission_started > 340 or pose is None):
            self.land_target = None
            self.phase_to("landing", "localisation_timeout")
            return

        if self.phase == "takeoff":
            if self.held(self.go(pose, HOME, ALTITUDE, 0.), .40):
                self.phase_to("search")
        elif self.phase == "search":
            if self.search_index >= len(self.search):
                self.phase_to("return", "search_exhausted")
                return
            xy, yaw = self.search[self.search_index]
            if self.held(self.go(pose, xy, ALTITUDE, yaw), 1.5):
                self.search_index += 1
                self.arrived = None
        elif self.phase == "inspect":
            track = self.active
            front = track.front
            yaw = math.atan2(-front[1], -front[0])
            xy = track.center + 1.78*front
            reached = self.go(pose, xy, ALTITUDE, yaw, speed=.65)
            ready = len(track.rows()) == 4
            if self.held(reached, 2.3 if ready else 8.0) or elapsed > 35:
                if len(self.tracks) == 3:
                    self.phase_to("overflight")
                else:
                    self.next_search()
        elif self.phase == "overflight":
            # After the last front view, reach the tower itself above the structure.
            front = self.active.front
            yaw = math.atan2(-front[1], -front[0])
            if self.held(self.go(pose, self.active.center, ALTITUDE, yaw, speed=.65), .5):
                count = sum(len(track.rows()) for track in self.tracks)
                self.phase_to("return", "" if count == 12 else "missing_insulators")
        elif self.phase == "return":
            if self.held(self.go(pose, HOME, ALTITUDE, 0.), .8):
                self.land_target = HOME
                self.phase_to("landing", self.reason)
        elif self.phase == "landing":
            if pose is not None and self.land_target is not None and pose.position[2] > .36:
                self.go(pose, self.land_target, .25, pose.yaw, speed=.28)
            else:
                self.send(z=-.23)
            # The down camera loses the map close to the floor; finish with slow descent.
            if elapsed > 12 or (pose is not None and pose.position[2] < .12 and elapsed > 3):
                self.send()
                self.enable.publish(Bool(data=False))
                phase = {"": "complete", "missing_insulators": "complete",
                         "operator_stop": "stopped"}.get(self.reason, "failed")
                self.phase_to(phase, self.reason)


def main():
    rclpy.init()
    node = Inspection()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.send()
            node.enable.publish(Bool(data=False))
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
