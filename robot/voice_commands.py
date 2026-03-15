import os
import time
import threading
import subprocess

import cv2
import numpy as np

from .constants import FACE_DATA_DIR
from .vision_detection import get_camera_frame_safe, capture_frame_pi

try:
    from cv2 import face as _cv2_face
    _LBPH_AVAILABLE = True
except (ImportError, AttributeError):
    _cv2_face = None
    _LBPH_AVAILABLE = False

FACE_CONFIDENCE_THRESHOLD = 60
FACE_MIN_SIZE = (80, 80)
FACE_NO_FACE_RESET_SEC = 5.0
FACE_LOOP_INTERVAL = 0.25
FACE_SIZE = (100, 100)


def _face_cascade_path():
    name = "haarcascade_frontalface_default.xml"
    paths = [
        os.path.join(FACE_DATA_DIR, name),
        "/usr/share/opencv4/data/haarcascades/" + name,
        "/usr/share/opencv/data/haarcascades/" + name,
        os.path.join(os.path.dirname(cv2.__file__), "data", name),
    ]
    cv2_data = getattr(cv2, "data", None)
    if cv2_data is not None:
        haarcascades = getattr(cv2_data, "haarcascades", None)
        if haarcascades:
            paths.insert(0, haarcascades + name)
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return paths[-1]


def _load_face_cascade():
    return cv2.CascadeClassifier(_face_cascade_path())


def _load_face_recognizer():
    if not _LBPH_AVAILABLE:
        return None, {}
    model_path = os.path.join(FACE_DATA_DIR, "lbph_model.yml")
    labels_path = os.path.join(FACE_DATA_DIR, "labels.txt")
    if not os.path.isfile(model_path) or not os.path.isfile(labels_path):
        return None, {}
    try:
        recognizer = _cv2_face.LBPHFaceRecognizer_create()
        recognizer.read(model_path)
        id_to_name = {}
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    id_to_name[int(parts[0])] = parts[1]
        return recognizer, id_to_name
    except Exception:
        return None, {}


def _detect_faces(gray, cascade):
    if cascade is None:
        return []
    return list(cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=FACE_MIN_SIZE, flags=cv2.CASCADE_SCALE_IMAGE
    ))


def _recognize_face(roi_gray, recognizer, id_to_name):
    if recognizer is None or not id_to_name:
        return None, 999
    try:
        label, confidence = recognizer.predict(roi_gray)
        if confidence > FACE_CONFIDENCE_THRESHOLD:
            return None, confidence
        return id_to_name.get(label), confidence
    except Exception:
        return None, 999


def robot_speak(text: str):
    if not text or not isinstance(text, str):
        return
    text = text.strip()
    if not text:
        return
    try:
        subprocess.Popen(["espeak", "-s", "140", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        try:
            subprocess.Popen(["espeak-ng", "-s", "140", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            pass
    except Exception:
        pass


class FaceRecognitionEngine:
    def __init__(self, speak_callback=None):
        self.speak_callback = speak_callback
        self.cascade = _load_face_cascade()
        self.recognizer, self.id_to_name = _load_face_recognizer()
        self._lock = threading.Lock()
        self._state = "no_face"
        self._last_face_time = 0.0

    def process_frame(self, frame_bgr):
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = _detect_faces(gray, self.cascade)
        results = []
        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            if roi.size == 0:
                continue
            roi = cv2.resize(roi, FACE_SIZE, interpolation=cv2.INTER_LINEAR)
            name, confidence = _recognize_face(roi, self.recognizer, self.id_to_name)
            results.append({"rect": (x, y, w, h), "name": name, "confidence": confidence})
        return results

    def update_greeting_state(self, results):
        if not self.speak_callback:
            return
        now = time.time()
        with self._lock:
            if not results:
                if self._state != "no_face" and (now - self._last_face_time) >= FACE_NO_FACE_RESET_SEC:
                    self._state = "no_face"
                return
            self._last_face_time = now
            if self._state == "no_face":
                known_name = None
                for r in results:
                    if r.get("name"):
                        known_name = r["name"]
                        break
                if known_name and known_name.strip().lower() == "sharath":
                    self.speak_callback("Hi Sharath, what's up?")
                    self._state = "greeted_known"
                else:
                    self.speak_callback("Hello there.")
                    self._state = "greeted_unknown"


def train_face_recognizer():
    if not _LBPH_AVAILABLE:
        return "opencv-contrib-python required for cv2.face"
    cascade = cv2.CascadeClassifier(_face_cascade_path())
    if cascade.empty():
        return "Could not load Haar cascade"
    faces_list, labels_list, id_to_name = [], [], {}
    current_id = 0
    if not os.path.isdir(FACE_DATA_DIR):
        os.makedirs(FACE_DATA_DIR, exist_ok=True)
        return "Created face_data. Add subfolders per person and retry."
    for name in sorted(os.listdir(FACE_DATA_DIR)):
        dir_path = os.path.join(FACE_DATA_DIR, name)
        if not os.path.isdir(dir_path) or name.startswith('.'):
            continue
        id_to_name[current_id] = name
        for fname in os.listdir(dir_path):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            path = os.path.join(dir_path, fname)
            img = cv2.imread(path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            for (x, y, w, h) in cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)):
                roi = cv2.resize(gray[y:y+h, x:x+w], FACE_SIZE, interpolation=cv2.INTER_LINEAR)
                faces_list.append(roi)
                labels_list.append(current_id)
        current_id += 1
    if not faces_list:
        return "No faces found. Add images to face_data/<name>/ and retry."
    os.makedirs(FACE_DATA_DIR, exist_ok=True)
    labels_path = os.path.join(FACE_DATA_DIR, "labels.txt")
    with open(labels_path, "w", encoding="utf-8") as f:
        for idx, name in sorted(id_to_name.items()):
            f.write(f"{idx} {name}\n")
    recognizer = _cv2_face.LBPHFaceRecognizer_create()
    recognizer.train(faces_list, np.array(labels_list))
    recognizer.save(os.path.join(FACE_DATA_DIR, "lbph_model.yml"))
    return f"Trained {len(faces_list)} faces. Restart app to use model."
