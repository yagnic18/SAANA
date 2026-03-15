# SAANA – Smart Autonomous AI Navigation Assistant

SAANA is a Raspberry Pi-based robotic assistant designed to perform basic autonomous navigation using sensors and computer vision. The system combines ultrasonic obstacle detection, camera-based vision processing, and motor control to allow a robot to move safely in its environment.

## Features

* Obstacle detection using ultrasonic sensors
* Computer vision processing using OpenCV
* Motor control using L298N driver
* Voice command module for basic interaction
* Flask-based API for controlling the robot

## Technologies Used

* Python
* OpenCV
* Flask
* Raspberry Pi GPIO
* Ultrasonic Sensors
* L298N Motor Driver

## Project Structure

```
SAANA
│
├── main.py
├── robot/        # Motor control and sensor logic
├── vision/       # Vision detection modules
├── models/       # AI models used for detection
├── face_data/    # Data used for recognition
├── requirements.txt
└── README.md
```

## How to Run

Install dependencies:

```
pip install -r requirements.txt
```

Run the main program:

```
python main.py
```

## Future Improvements

* Improve object detection accuracy
* Add better voice interaction system
* Deploy full autonomous navigation
