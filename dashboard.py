#!/usr/bin/env python3
"""Serve the local operator interface and forward commands to ROS services."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
from threading import Event, Lock, Thread
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String
from std_srvs.srv import Trigger

ROOT = Path(__file__).resolve().parent
ACTIONS = ("start", "stop", "kill")


class Dashboard(Node):
    def __init__(self, generated):
        super().__init__("lep_dashboard")
        self.generated = generated
        reference = json.loads((generated / "reference.json").read_text())
        self.map = json.loads((generated / "aruco_map.json").read_text())
        self.map["wires"] = reference["wires"]
        self.mission = None
        self.start_lock = Lock()
        self.cancel_start = Event()
        self.status = {"phase": "idle", "reason": "", "found": 0}
        self.results = {"frame_id": "aruco_map", "insulators": []}
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/insulators", self.on_results, qos)
        self.create_subscription(String, "/inspection/status", self.on_status, qos)
        self.mission_services = {action: self.create_client(Trigger, f"/inspection/{action}")
                         for action in ACTIONS}

    def on_results(self, msg):
        self.results = json.loads(msg.data)

    def on_status(self, msg):
        self.status = json.loads(msg.data)

    def command(self, action):
        client = self.mission_services[action]
        if action == "start":
            if not self.start_lock.acquire(blocking=False):
                return False, "Запуск уже выполняется."
            try:
                self.cancel_start.clear()
                if not client.service_is_ready() and (
                        self.mission is None or self.mission.poll() is not None):
                    script = ROOT / "mission.py"
                    self.get_logger().info(f"Запуск {script.name}")
                    self.mission = subprocess.Popen([
                        sys.executable, str(script), "--ros-args",
                        "-p", f"marker_map:={self.generated / 'aruco_map.json'}",
                        "-p", "use_sim_time:=true",
                    ])
                end = time.monotonic()+16
                while time.monotonic() < end and not self.cancel_start.is_set():
                    if client.wait_for_service(timeout_sec=.4):
                        if self.cancel_start.is_set():
                            break
                        ok, message = self.invoke(client)
                        if ok or not message.startswith("Ожидаю"):
                            return ok, message
                        time.sleep(.35)
                if self.cancel_start.is_set():
                    return False, "Запуск отменён."
                return False, "Нет связи с симулятором или видеопотоками."
            finally:
                self.start_lock.release()
        if action == "kill" and self.start_lock.locked():
            self.cancel_start.set()
            if not client.service_is_ready():
                return True, "Запуск отменён."
        return self.invoke(client) if client.service_is_ready() else (
            False, "ROS-нода обследования ещё не запущена.")

    @staticmethod
    def invoke(client):
        event = Event()
        future = client.call_async(Trigger.Request())
        future.add_done_callback(lambda _: event.set())
        if not event.wait(timeout=3):
            return False, "ROS-сервис не ответил."
        try:
            reply = future.result()
        except Exception:
            return False, "Ошибка вызова ROS-сервиса."
        return reply.success, reply.message


def serve(node, host, port):
    page = (ROOT / "web" / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def send(self, body, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def json(self, value, status=200):
            self.send(json.dumps(value, ensure_ascii=False).encode(),
                      "application/json; charset=utf-8", status)

        def do_GET(self):
            if self.path == "/":
                self.send(page, "text/html; charset=utf-8")
            elif self.path == "/map":
                self.json(node.map)
            elif self.path == "/state":
                self.json({"status": node.status, "results": node.results})
            else:
                self.send_error(404)

        def do_POST(self):
            action = self.path.removeprefix("/command/")
            if self.path != f"/command/{action}" or action not in ACTIONS:
                self.send_error(404)
                return
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                self.send_error(403)
                return
            ok, message = node.command(action)
            self.json({"ok": ok, "message": message}, 200 if ok else 409)

        def log_message(self, *_):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, default=ROOT / "generated")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    rclpy.init()
    node = Dashboard(args.generated.resolve())
    server = serve(node, "127.0.0.1", args.port)
    http = Thread(target=server.serve_forever, daemon=True)
    http.start()
    print(f"Панель оператора: http://127.0.0.1:{args.port}", flush=True)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        server.shutdown()
        server.server_close()
        if node.mission and node.mission.poll() is None:
            node.mission.terminate()
            try:
                node.mission.wait(timeout=5)
            except subprocess.TimeoutExpired:
                node.mission.kill()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
