import base64
import time
import threading

import cv2
from flask import Flask, Response, jsonify, render_template_string, request
from .constants import CRITICAL_CM, WARNING_CM, SAFE_CM
from .robot_state import set_critical_stop, get_critical_stop
from .motor_control import MotorController
from .ultrasonic_sensor import UltrasonicSystem
from .vision_detection import (
    capture_frame_pi,
    local_detect_objects,
    local_scene_description,
    camera_depth_approximation,
    detect_table_edge,
    get_camera_frame_safe,
    RpiCamCamera,
)
from .voice_commands import robot_speak, FaceRecognitionEngine, train_face_recognizer

app = Flask(__name__)
HTML_TEMPLATE = '<html><body><h1>Sana Robot</h1><img src="/video_feed"></body></html>'

motors = None
sensors = None
global_camera = None
face_engine = None

safety_monitor = None

class SafetyMonitor:
    def __init__(self, motors_ref, sensors_ref, camera_ref):
        self.motors = motors_ref
        self.sensors = sensors_ref
        self.camera = camera_ref
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)

    def _run(self):
        while self.running:
            critical = False
            if self.sensors and getattr(self.sensors, 'enabled', False):
                d = self.sensors.get_all_distances()
                if any(v >= 0 and v < CRITICAL_CM for v in d.values()):
                    critical = True
            if self.camera:
                frame = get_camera_frame_safe(self.camera)
                if frame is not None:
                    zone, _ = camera_depth_approximation(frame)
                    if zone == 'critical':
                        critical = True
                    if detect_table_edge(frame):
                        critical = True
            if critical:
                set_critical_stop(True)
                if self.motors:
                    self.motors.stop()
            else:
                set_critical_stop(False)
            time.sleep(0.05)


def init_hardware():
    global motors, sensors, global_camera, safety_monitor, face_engine
    try:
        motors = MotorController()
    except Exception as e:
        print('Motor init failed', e)
        motors = None
    try:
        sensors = UltrasonicSystem()
    except Exception as e:
        print('Sensor init failed', e)
        sensors = None
    try:
        global_camera = RpiCamCamera(width=640, height=480)
    except Exception as e:
        print('Camera init failed', e)
        global_camera = None
    if motors and (sensors or global_camera):
        safety_monitor = SafetyMonitor(motors, sensors, global_camera)
        safety_monitor.start()
    if global_camera:
        face_engine = FaceRecognitionEngine(speak_callback=robot_speak)
        t = threading.Thread(target=_face_recognition_loop, daemon=True)
        t.start()


def _face_recognition_loop():
    global face_engine
    while getattr(face_engine, '_running', True) and global_camera is not None:
        frame = get_camera_frame_safe(global_camera)
        if frame is not None:
            results = face_engine.process_frame(frame)
            face_engine.update_greeting_state(results)
        time.sleep(0.25)


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/move/<direction>')
def move(direction):
    if not motors:
        return jsonify({'status': 'error', 'message': 'Hardware not initialized'})
    if direction == 'forward':
        motors.forward()
    elif direction == 'backward':
        motors.backward()
    elif direction == 'left':
        motors.left()
    elif direction == 'right':
        motors.right()
    elif direction == 'stop':
        motors.stop()
    return jsonify({'status': 'success', 'action': direction})


@app.route('/detect')
def detect():
    frame = capture_frame_pi()
    if frame is None:
        return jsonify({'error': 'could not capture frame'})
    det = local_detect_objects(frame)
    return jsonify({'result': local_scene_description(det)})
