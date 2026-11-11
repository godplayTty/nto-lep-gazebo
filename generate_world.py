#!/usr/bin/env python3
"""Write the Gazebo world, navigation marker map, and reference JSON."""
import argparse
import json
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import cv2

from geometry import (CAMERAS, INSULATOR_LENGTH, PLATE_HEIGHT, PLATE_OFFSET,
                      PLATE_SIZE, POSITIONS, STATES, WORLD, marker_map)

STEEL = (0.47, 0.50, 0.54)
DARK = (0.26, 0.28, 0.31)
BLUE = (0.015, 0.15, 0.94)
CERAMIC = {"black": (0.013, 0.015, 0.017), "brown": (0.43, 0.16, 0.052),
           "red": (0.87, 0.025, 0.032)}
TOWER_HEIGHT = 1.50
# The foot plates and steel thickness count toward the outside dimensions.
BASE_HALF = .286
TOP_HALF = .080
FOOT_WIDTH = .052


def node(parent, tag, value=None, **attrs):
    child = ET.SubElement(parent, tag, attrs)
    if isinstance(value, (list, tuple)):
        value = " ".join(map(str, value))
    if value is not None:
        child.text = str(value)
    return child


def save_xml(tree, path):
    ET.indent(tree, space="  ")
    ET.ElementTree(tree).write(path, encoding="utf-8", xml_declaration=True)


def appearance(visual, color, emissive=False):
    material = node(visual, "material")
    node(material, "ambient", (*color, 1))
    node(material, "diffuse", (*color, 1))
    node(material, "specular", (0.04, 0.04, 0.04, 1))
    if emissive:
        node(material, "lighting", "false")
        node(material, "emissive", (*color, 1))


def shape(link, name, pose, primitive, size, color, collision=False, emissive=False):
    visual = node(link, "visual", name=name)
    node(visual, "pose", pose)
    entities = [visual]
    if collision:
        solid = node(link, "collision", name=name)
        node(solid, "pose", pose)
        entities.append(solid)
    for entity in entities:
        geometry = node(node(entity, "geometry"), primitive)
        if primitive == "box":
            node(geometry, "size", size)
        else:
            node(geometry, "radius", size[0])
            node(geometry, "length", size[1])
    appearance(visual, color, emissive)


def box(link, name, center, size, color=STEEL, **options):
    shape(link, name, (*center, 0, 0, 0), "box", size, color, **options)


def beam(link, name, a, b, width, color=STEEL, cylinder=False):
    dx, dy, dz = (end-start for start, end in zip(a, b))
    length = math.sqrt(dx*dx + dy*dy + dz*dz)
    pose = (*(0.5*(start+end) for start, end in zip(a, b)), 0,
            math.atan2(math.hypot(dx, dy), dz), math.atan2(dy, dx))
    if cylinder:
        shape(link, name, pose, "cylinder", (width/2, length), color)
    else:
        shape(link, name, pose, "box", (width, width, length), color)


def static_link(world, name, pose=(0, 0, 0, 0, 0, 0)):
    model = node(world, "model", name=name)
    node(model, "static", "true")
    node(model, "pose", pose)
    return node(model, "link", name="structure")


def tower(world, name, x, y, yaw):
    link = static_link(world, name, (x, y, 0, 0, 0, yaw))
    signs = [(1, 1), (-1, 1), (-1, -1), (1, -1)]

    def half_width(z):
        return BASE_HALF + (TOP_HALF - BASE_HALF) * z / TOWER_HEIGHT

    def corner(i, z):
        half = half_width(z)
        return signs[i][0]*half, signs[i][1]*half, z

    for i in range(4):
        beam(link, f"leg_{i}", corner(i, 0.018), corner(i, 1.485), 0.028)
        foot = corner(i, 0)
        box(link, f"base_plate_{i}", (*foot[:2], 0.009), (FOOT_WIDTH, FOOT_WIDTH, .018), DARK,
            collision=True)
        for k, (dx, dy) in enumerate([(-.020, -.020), (-.020, .020), (.020, -.020), (.020, .020)]):
            shape(link, f"bolt_{i}_{k}", (foot[0]+dx, foot[1]+dy, .022, 0, 0, 0),
                  "cylinder", (.004, .009), STEEL)
        j = (i+1) % 4
        levels = (.05, .37, .70, 1.08, 1.485)
        for k, z in enumerate(levels):
            beam(link, f"tie_{i}_{k}", corner(i, z), corner(j, z), 0.019)
        for k, (a, b) in enumerate(zip(levels, levels[1:])):
            beam(link, f"brace_{i}_{k}_a", corner(i, a), corner(j, b), 0.014)
            beam(link, f"brace_{i}_{k}_b", corner(j, a), corner(i, b), 0.014)

    for number, (side, z) in POSITIONS.items():
        sign = math.copysign(1, side)
        root = sign*half_width(z)
        lower = sign*half_width(z-.16)
        for j, direction in enumerate((-1, 1)):
            beam(link, f"arm_{number}_{j}", (direction*abs(root), root, z),
                 (0, side, z), .024)
            beam(link, f"strut_{number}_{j}", (direction*abs(lower), lower, z-.16),
                 (0, side, z), .017)
            beam(link, f"arm_brace_{number}_{j}", (direction*abs(lower)/2, (lower+side)/2, z-.08),
                 (direction*abs(root)/2, (root+side)/2, z), .013)
            box(link, f"gusset_{number}_{j}", (direction*abs(root), root, z),
                (.052, .042, .064), DARK)
        box(link, f"hanger_{number}", (0, side, z-.011), (.036, .032, .025), DARK)
    beam(link, "sign_mount", (.221, -.217, PLATE_HEIGHT),
         (.221, .217, PLATE_HEIGHT), .023)
    box(link, "sign_back", (.229, 0, PLATE_HEIGHT), (.015, PLATE_SIZE, PLATE_SIZE), DARK)
    # A front-facing plane has no blue surface when seen from the rear.
    face = node(link, "visual", name="sign_blue")
    node(face, "pose", (PLATE_OFFSET, 0, PLATE_HEIGHT, 0, math.pi/2, 0))
    plane = node(node(face, "geometry"), "plane")
    node(plane, "normal", (0, 0, 1))
    node(plane, "size", (PLATE_SIZE, PLATE_SIZE))
    appearance(face, BLUE, emissive=True)
    for i, (dy, dz) in enumerate([(-.08, -.08), (-.08, .08), (.08, -.08), (.08, .08)]):
        shape(link, f"sign_bolt_{i}", (PLATE_OFFSET+.001, dy, PLATE_HEIGHT+dz, 0, math.pi/2, 0),
              "cylinder", (.004, .003), STEEL)


def insulator(world, name, x, y, z, color):
    link = static_link(world, name, (x, y, z, 0, 0, 0))
    shape(link, "core", (0, 0, -.070, 0, 0, 0), "cylinder", (.022, .120), CERAMIC[color])
    for index in range(6):
        shape(link, f"skirt_{index}", (0, 0, -.024-.020*index, 0, 0, 0),
              "cylinder", (.050 if index % 2 == 0 else .043, .015), CERAMIC[color])
    shape(link, "top_fitting", (0, 0, -.009, 0, 0, 0), "cylinder", (.027, .018), STEEL)
    shape(link, "bottom_fitting", (0, 0, -.132, 0, 0, 0), "cylinder", (.027, .014), STEEL)
    shape(link, "eye", (0, 0, -.142, 0, 0, 0), "cylinder",
          (.014, 2 * (INSULATOR_LENGTH - .142)), DARK)


def marker_mesh(path, marks):
    # Separate black and white meshes share the same geometry convention as OpenCV.
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    colors = {"black": [], "white": []}

    def quad(color, left, top, right, bottom, z):
        colors[color].extend([(left, bottom, z), (right, bottom, z),
                              (right, top, z), (left, top, z)])

    for mark in marks:
        x, y, z, size = (mark[key] for key in ("x", "y", "z", "size"))
        quad("white", x-.23, y+.23, x+.23, y-.23, z-.001)
        if hasattr(cv2.aruco, "generateImageMarker"):
            pixels = cv2.aruco.generateImageMarker(dictionary, mark["id"], 8)
        else:
            pixels = cv2.aruco.drawMarker(dictionary, mark["id"], 8)
        step = size / 8
        for row in range(8):
            for col in range(8):
                if not pixels[row, col]:
                    left = x - size/2 + step*col
                    top = y + size/2 - step*row
                    quad("black", left, top, left+step, top-step, z)

    for color, vertices in colors.items():
        with (path / f"markers_{color}.obj").open("w") as mesh:
            mesh.write("vn 0 0 1\n")
            for p in vertices:
                mesh.write("v %.6f %.6f %.6f\n" % p)
            for i in range(1, len(vertices)+1, 4):
                mesh.write(f"f {i}//1 {i+1}//1 {i+2}//1\nf {i}//1 {i+2}//1 {i+3}//1\n")


def plugin(parent, filename, classname, **settings):
    element = node(parent, "plugin", name=f"gz::sim::systems::{classname}",
                   filename=f"gz-sim-{filename}-system")
    for key, value in settings.items():
        node(element, key, value)
    return element


def inertial(link, mass, ixx, iyy, izz):
    data = node(link, "inertial")
    node(data, "mass", mass)
    inertia = node(data, "inertia")
    for key, value in (("ixx", ixx), ("iyy", iyy), ("izz", izz)):
        node(inertia, key, value)


def quadcopter(world):
    model = node(world, "model", name="inspection_quad")
    node(model, "pose", (0, 0, .080, 0, 0, 0))
    body = node(model, "link", name="base_link")
    inertial(body, .85, .017, .017, .028)
    box(body, "chassis", (0, 0, 0), (.24, .17, .07), DARK, collision=True)
    box(body, "front_camera_housing", (.106, 0, -.021), (.045, .045, .039), DARK)
    for side in (-1, 1):
        box(body, f"landing_skid_{side}", (0, side*.071, -.066), (.24, .018, .025),
            DARK, collision=True)
        for index, x in enumerate((-.075, .075)):
            box(body, f"landing_leg_{side}_{index}", (x, side*.071, -.045),
                (.018, .018, .055), STEEL)

    controller = plugin(model, "multicopter-control", "MulticopterVelocityControl",
                        robotNamespace="inspection_quad", commandSubTopic="cmd_vel",
                        enableSubTopic="enable", comLinkName="base_link",
                        velocityGain=(2.7, 2.7, 3.5), attitudeGain=(2.4, 2.4, .35),
                        angularRateGain=(.45, .45, .22), maximumLinearAcceleration=(2, 2, 3))
    config = node(controller, "rotorConfiguration")
    for index, (x, y) in enumerate(((.18, .18), (-.18, .18), (-.18, -.18), (.18, -.18))):
        rotor_name = f"rotor_{index}"
        joint_name = f"joint_{index}"
        beam(body, f"arm_{index}", (0, 0, 0), (x, y, .01), .025,
             (0.12, .42, .24) if x > 0 else DARK)
        rotor = node(model, "link", name=rotor_name)
        node(rotor, "pose", (x, y, .027, 0, 0, 0))
        inertial(rotor, .008, .000018, .000018, .000034)
        box(rotor, "propeller", (0, 0, 0), (.235, .018, .006), DARK)
        joint = node(model, "joint", name=joint_name, type="revolute")
        node(joint, "parent", "base_link")
        node(joint, "child", rotor_name)
        node(node(joint, "axis"), "xyz", (0, 0, 1))
        direction = 1 if index % 2 == 0 else -1
        plugin(model, "multicopter-motor-model", "MulticopterMotorModel",
               robotNamespace="inspection_quad", jointName=joint_name,
               linkName=rotor_name, turningDirection="ccw" if direction == 1 else "cw",
               timeConstantUp=.0125, timeConstantDown=.025, maxRotVelocity=800,
               motorConstant=8.54858e-6, momentConstant=.016,
               commandSubTopic="motor_speed", actuator_number=index,
               rotorDragCoefficient=8.06428e-5, rollingMomentCoefficient=1e-6,
               motorSpeedPubTopic=f"speed/{index}", rotorVelocitySlowdownSim=10,
               motorType="velocity")
        element = node(config, "rotor")
        node(element, "jointName", joint_name)
        node(element, "forceConstant", 8.54858e-6)
        node(element, "momentConstant", .016)
        node(element, "direction", direction)

    for name, spec in CAMERAS.items():
        sensor = node(body, "sensor", name=f"{name}_camera", type="camera")
        node(sensor, "pose", (*spec["translation"], 0, spec["pitch"], 0))
        node(sensor, "topic", f"/camera/{name}/image")
        node(sensor, "gz_frame_id", f"{name}_optical")
        node(sensor, "always_on", "true")
        node(sensor, "update_rate", 12 if name == "down" else 8)
        camera = node(sensor, "camera")
        node(camera, "camera_info_topic", f"/camera/{name}/camera_info")
        node(camera, "horizontal_fov", spec["fov"])
        image = node(camera, "image")
        node(image, "width", spec["width"])
        node(image, "height", spec["height"])
        node(image, "format", "R8G8B8")
        clip = node(camera, "clip")
        node(clip, "near", .018)
        node(clip, "far", 38)


def generate(path, seed):
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    randomizer = random.Random(seed)
    root = ET.Element("sdf", version="1.10")
    world = node(root, "world", name=WORLD)
    physics = node(world, "physics", name="physics", type="ignored")
    node(physics, "max_step_size", .004)
    node(physics, "real_time_factor", 1)
    for filename, name in (("physics", "Physics"), ("scene-broadcaster", "SceneBroadcaster"),
                           ("user-commands", "UserCommands")):
        plugin(world, filename, name)
    plugin(world, "sensors", "Sensors", render_engine="ogre2")
    scene = node(world, "scene")
    node(scene, "ambient", (.60, .60, .60, 1))
    node(scene, "background", (.73, .81, .89, 1))
    node(scene, "shadows", "false")
    light = node(world, "light", type="directional", name="sun")
    node(light, "pose", (0, 0, 10, 0, 0, 0))
    node(light, "diffuse", (.68, .68, .68, 1))
    node(light, "specular", (.14, .14, .14, 1))
    node(light, "direction", (.22, -.38, -.9))
    box(static_link(world, "ground"), "ground", (6, 0, -.025),
        (38, 30, .05), (.59, .64, .60), collision=True)
    marks = marker_map()
    marker_mesh(path, marks)
    floor = static_link(world, "aruco_floor")
    for color, rgb in (("black", (.008, .008, .008)), ("white", (.99, .99, .99))):
        visual = node(floor, "visual", name=f"markers_{color}")
        node(node(node(visual, "geometry"), "mesh"), "uri",
             (path / f"markers_{color}.obj").as_uri())
        appearance(visual, rgb, emissive=True)
    quadcopter(world)

    heading = randomizer.uniform(-.40, .40)
    travel = math.cos(heading), math.sin(heading)
    facing = (-travel[0], -travel[1])
    right = -facing[1], facing[0]
    origin = (randomizer.uniform(2.65, 3.12), randomizer.uniform(-.18, .18))
    length = randomizer.uniform(5., 9.8)
    distances = (0., randomizer.uniform(2.0, length-2.0), length)
    while True:
        colors = randomizer.choices(list(STATES), k=12)
        if set(colors) == set(STATES):
            break
    reference = {"seed": seed, "frame_id": "aruco_map", "towers": [], "wires": []}
    ends = []
    for i, distance in enumerate(distances):
        x, y = (a+distance*b for a, b in zip(origin, travel))
        name = f"tower_{i+1}"
        tower(world, name, x, y, heading + math.pi)
        record = {"id": name, "x": x, "y": y, "front": facing, "insulators": []}
        connection = {}
        for number, (side, z) in POSITIONS.items():
            ix, iy = x+side*right[0], y+side*right[1]
            color = colors[4*i + number-1]
            insulator(world, f"{name}_insulator_{number}", ix, iy, z, color)
            record["insulators"].append({"number": number, "color": color, "state": STATES[color]})
            connection[number] = (ix, iy, z-INSULATOR_LENGTH)
        reference["towers"].append(record)
        ends.append(connection)

    for i, (a, b) in enumerate(zip(ends, ends[1:]), 1):
        span = static_link(world, f"span_{i}")
        for number in POSITIONS:
            beam(span, f"wire_{number}", a[number], b[number], .008, DARK, cylinder=True)
            reference["wires"].append({"span": i, "number": number,
                                       "from": a[number], "to": b[number]})

    save_xml(root, path / "lep.sdf")
    for name, contents in (("aruco_map.json", {"frame_id": "aruco_map", "markers": marks}),
                           ("reference.json", reference)):
        (path / name).write_text(json.dumps(contents, ensure_ascii=False, indent=2)+"\n")
    return path / "lep.sdf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("generated"))
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    print(f"seed={seed} world={generate(args.out, seed)}")


if __name__ == "__main__":
    main()
