import os
import time
import subprocess
import threading
import base64

import cv2
import numpy as np

from .constants import PI_CAM_WIDTH, PI_CAM_HEIGHT, MODELS_DIR, VISION_DIR


# --- Vision capture helpers ---

def capture_frame_opencv():
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, PI_CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PI_CAM_HEIGHT)
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            return frame
    except Exception as e:
        print(f"OpenCV capture error: {e}")
    finally:
        if cap is not None:
            cap.release()
    return None


def capture_frame_pi():
    for cmd, out_flag in [("rpicam-still", "-o"), ("libcamera-still", "-o")]:
        try:
            result = subprocess.run(
                [cmd, "-t", "1", "--width", str(PI_CAM_WIDTH), "--height", str(PI_CAM_HEIGHT), out_flag, "-", "-n"],
                capture_output=True, timeout=10
            )
            if result.returncode == 0 and result.stdout:
                img = cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    return img
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"{cmd} error: {e}")
    for cmd, out_arg in [("rpicam-vid", ["--output", "-"]), ("libcamera-vid", ["-o", "-"])]:
        try:
            proc = subprocess.Popen(
                [cmd, "-t", "2000", "--codec", "mjpeg", "--width", str(PI_CAM_WIDTH), "--height", str(PI_CAM_HEIGHT), "--inline", "--nopreview"] + out_arg,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
            )
            data, deadline = b"", time.time() + 5
            while time.time() < deadline:
                chunk = proc.stdout.read(8192)
                if not chunk:
                    time.sleep(0.05)
                    continue
                data += chunk
                start = data.find(b"\xff\xd8")
                end = data.find(b"\xff\xd9")
                if start != -1 and end != -1 and end > start:
                    jpg = data[start:end + 2]
                    proc.terminate()
                    proc.wait(timeout=2)
                    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        return img
                    break
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"{cmd} stream error: {e}")
    frame = capture_frame_opencv()
    return frame


def preprocess_vision_frame_opencv(image_bgr):
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    max_side = 1280
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    return image_bgr


_LOCAL_DETECT_CONFIDENCE = 0.45
_LOCAL_DETECT_MODEL = None
_LOCAL_DETECT_LOCK = threading.Lock()


def _load_local_detector():
    global _LOCAL_DETECT_MODEL
    with _LOCAL_DETECT_LOCK:
        if _LOCAL_DETECT_MODEL is not None:
            return True
        try:
            from ultralytics import YOLO
            _LOCAL_DETECT_MODEL = YOLO('yolov8n.pt')
            print("Local vision: YOLOv8 loaded")
            return True
        except ImportError:
            print("Local vision: ultralytics not installed")
            return False
        except Exception as e:
            print(f"Local vision: could not load YOLOv8: {e}")
            return False


def _detect_objects_contour_fallback(frame_bgr):
    if frame_bgr is None or frame_bgr.size == 0:
        return []
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        cx = x + bw // 2
        if cx < w * 0.33:
            pos = "left"
        elif cx > w * 0.66:
            pos = "right"
        else:
            pos = "center"
        out.append(("object", pos))
    return out


def local_detect_objects(frame_bgr):
    if frame_bgr is None or frame_bgr.size == 0:
        return []
    h, w = frame_bgr.shape[:2]
    if not _load_local_detector():
        return _detect_objects_contour_fallback(frame_bgr)
    try:
        out = []
        with _LOCAL_DETECT_LOCK:
            if _LOCAL_DETECT_MODEL is None:
                return _detect_objects_contour_fallback(frame_bgr)
            results = _LOCAL_DETECT_MODEL(frame_bgr, conf=_LOCAL_DETECT_CONFIDENCE, verbose=False)
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf)
                if conf < _LOCAL_DETECT_CONFIDENCE:
                    continue
                class_id = int(box.cls)
                class_name = _LOCAL_DETECT_MODEL.names[class_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = int((x1 + x2) / 2)
                if cx < w * 0.33:
                    pos = "left"
                elif cx > w * 0.66:
                    pos = "right"
                else:
                    pos = "center"
                out.append((class_name, pos))
        return out
    except Exception as e:
        print(f"YOLOv8 error: {e}")
        return _detect_objects_contour_fallback(frame_bgr)


def local_scene_description(detections):
    if not detections:
        return "The area looks clear."
    seen = set()
    phrases = []
    for name, pos in detections:
        key = (name, pos)
        if key in seen:
            continue
        seen.add(key)
        phrases.append(f"{name} on the {pos}")
    if not phrases:
        return "The area looks clear."
    return "I see " + ", ".join(phrases[:5]) + "."


def vision_to_spoken(desc):
    if not desc or not isinstance(desc, str):
        return "The area looks clear."
    s = desc.strip()
    if not s or s == "The area looks clear.":
        return "The area looks clear."
    return s


def camera_depth_approximation(frame_bgr):
    if frame_bgr is None or frame_bgr.size == 0:
        return "clear", -1
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_dist = -1
    closest_zone = "clear"
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1500:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        cy = y + bh // 2
        area_score = area / (w * h)
        vert_score = 1.0 - (cy / h)
        closeness = area_score * 2.0 + vert_score
        if closeness > 1.5:
            approx_cm = max(10, 100 - closeness * 40)
        elif closeness > 0.8:
            approx_cm = max(20, 150 - closeness * 80)
        else:
            approx_cm = 200
        if min_dist < 0 or approx_cm < min_dist:
            min_dist = approx_cm
            if approx_cm < 25:
                closest_zone = "critical"
            elif approx_cm < 45:
                closest_zone = "warning"
            elif approx_cm < 80:
                closest_zone = "safe"
    if min_dist < 0:
        return "clear", -1
    return closest_zone, min_dist


def detect_table_edge(frame_bgr):
    if frame_bgr is None or frame_bgr.size == 0:
        return False
    h, w = frame_bgr.shape[:2]
    bottom_strip = frame_bgr[int(h * 0.65):, :]
    gray = cv2.cvtColor(bottom_strip, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=w * 0.3, maxLineGap=20)
    if lines is None:
        return False
    horizontal_count = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 15 or angle > 165:
            horizontal_count += 1
            if horizontal_count >= 2:
                return True
    return False


def get_camera_frame_safe(camera_obj):
    if camera_obj is None:
        return None
    try:
        if camera_obj.mode == "capture":
            return camera_obj._capture_single_frame()
        with camera_obj.lock:
            return camera_obj.frame.copy() if camera_obj.frame is not None else None
    except Exception:
        return None


class RpiCamCamera:
    """Camera helper that tries rpicam/libcamera streaming, then capture fallback."""
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.process = None
        self.mode = None

        if self._try_rpicam_vid():
            self.mode = 'rpicam'
        elif self._try_libcamera_vid():
            self.mode = 'libcamera'
        else:
            self.mode = 'capture'

        if self.process:
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()

    def _try_rpicam_vid(self):
        try:
            self.process = subprocess.Popen(
                ["rpicam-vid", "-t", "0", "--codec", "mjpeg", "--width", str(self.width), "--height", str(self.height), "--inline", "--nopreview", "--output", "-"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
            )
            time.sleep(1)
            return self.process.poll() is None
        except Exception:
            return False

    def _try_libcamera_vid(self):
        try:
            self.process = subprocess.Popen(
                ["libcamera-vid", "-t", "0", "--codec", "mjpeg", "--width", str(self.width), "--height", str(self.height), "--inline", "--nopreview", "-o", "-"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
            )
            time.sleep(1)
            return self.process.poll() is None
        except Exception:
            return False

    def _reader(self):
        data = b""
        while self.running and self.process:
            try:
                chunk = self.process.stdout.read(4096)
                if not chunk:
                    time.sleep(0.01)
                    continue
                data += chunk
                start = data.find(b"\xff\xd8")
                end = data.find(b"\xff\xd9")
                if start != -1 and end != -1 and end > start:
                    jpg = data[start:end + 2]
                    data = data[end + 2:]
                    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        with self.lock:
                            self.frame = img
            except Exception as e:
                print(f"Camera reader error: {e}")
                break

    def _capture_single_frame(self):
        for cmd in ["rpicam-still", "libcamera-still"]:
            try:
                result = subprocess.run(
                    [cmd, "-t", "1", "--width", str(self.width), "--height", str(self.height), "-o", "-", "-n"],
                    capture_output=True, timeout=5
                )
                if result.returncode == 0 and result.stdout:
                    img = cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        return img
            except Exception:
                continue
        return capture_frame_opencv()

    def get_frame(self):
        if self.mode == 'capture':
            img = self._capture_single_frame()
            if img is not None:
                _, jpeg = cv2.imencode('.jpg', img)
                return jpeg.tobytes()
            return None
        with self.lock:
            if self.frame is None:
                img = self._capture_single_frame()
                if img is not None:
                    _, jpeg = cv2.imencode('.jpg', img)
                    return jpeg.tobytes()
                return None
            _, jpeg = cv2.imencode('.jpg', self.frame)
            return jpeg.tobytes()

    def get_base64_frame(self):
        if self.mode == 'capture':
            img = self._capture_single_frame()
            if img is not None:
                _, buf = cv2.imencode('.jpg', img)
                return base64.b64encode(buf).decode('utf-8')
            return None
        with self.lock:
            if self.frame is None:
                img = self._capture_single_frame()
                if img is not None:
                    _, buf = cv2.imencode('.jpg', img)
                    return base64.b64encode(buf).decode('utf-8')
                return None
            _, buf = cv2.imencode('.jpg', self.frame)
            return base64.b64encode(buf).decode('utf-8')

    def stop(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=2)
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
