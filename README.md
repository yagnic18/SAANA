# SAANA Raspberry Pi AI Robot

A Python-based Raspberry Pi AI robot platform with obstacle avoidance, camera-based object/face detection, voice notifications, and Flask web controls.

## 🚀 Project Summary

This project integrates:
- Motor control via `gpiozero`.
- Ultrasonic distance sensors for safety.
- Raspberry Pi camera vision with OpenCV and YOLOv8 fallback.
- Face recognition (LBPH) and speech output via `espeak`.
- Web interface served by Flask for remote movement and detection.

## 📁 Repository Structure

```
SAANA/
├── face_data/               # Face training data per person
├── models/                  # Downloaded vision/model files
├── robot/
│   ├── __init__.py
│   ├── api_server.py        # Flask API + safety monitor + hardware init
│   ├── constants.py         # GPIO pins, thresholds, directories
│   ├── motor_control.py     # Motor driving interface
│   ├── robot_state.py       # Shared critical-stop state
│   ├── ultrasonic_sensor.py # Distance sensor system
│   ├── vision_detection.py  # Camera capture, YOLO detection, depth/edge analysis
│   └── voice_commands.py    # Face recognition, speech output, training helper
├── vision/                  # Optional vision assets, logs, config
├── main.py                  # Entrypoint to start Flask app
├── requirements.txt         # Python dependencies
└── README.md                # Project docs
```

## ✅ Features

- Web-based control endpoints: `/move/<direction>`, `/detect`
- Real-time safety monitoring across ultrasonic sensors & camera obstacles
- Automatic critical stop and motor shutoff upon proximity warnings
- Face recognition integration with greeting voice responses
- Modular hardware initialization for relaxed startup on missing components

## ⚙️ Requirements

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

### Hardware

- Raspberry Pi (4 recommended)
- Motor driver and two DC motors
- Ultrasonic distance sensors (HC-SR04)
- Raspberry Pi camera module (or USB webcam)
- Speaker or headphone output for `espeak`

### Software

- Python 3.9+ (3.11 recommended)
- Flask web server
- OpenCV with contrib modules for face recognition
- YOLOv8 (via `ultralytics`)

## ▶️ Run the Robot

1. Ensure your GPIO pins are configured correctly in `robot/constants.py`.
2. Start the app:

```bash
python main.py
```

3. Open your browser at `http://<pi-ip>:5000`.

## 🧠 Notes

- The code is intentionally modular and does not change runtime behavior automatically.
- `robot/vision_detection.py` uses `rpicam-still`/`libcamera-still` and streams if available.
- Face recognition uses LBPH and requires `opencv-contrib-python`.

## 📌 Quick API

- `GET /` – Simple status page with camera preview placeholder.
- `GET /move/<forward|backward|left|right|stop>` – Drive commands.
- `GET /detect` – Captures frame and returns scene detection summary.

## 📝 Contribution

1. Fork the repo.
2. Create a feature branch.
3. Open PR with hardware/test details.

---

If you want, I can also add a `setup.sh` for Raspberry Pi package install steps and a minimal `docker`/`systemd` service file.
