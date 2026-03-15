import time

from gpiozero import DistanceSensor
from .constants import (TRIG_FRONT_LEFT, ECHO_FRONT_LEFT, TRIG_FRONT_RIGHT,
                        ECHO_FRONT_RIGHT, TRIG_BACK, ECHO_BACK)


def _read_hcsr04_manual(trigger_gpio, echo_gpio):
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(trigger_gpio, GPIO.OUT)
        GPIO.setup(echo_gpio, GPIO.IN)
        GPIO.output(trigger_gpio, False)
        time.sleep(0.00002)
        GPIO.output(trigger_gpio, True)
        time.sleep(0.00001)
        GPIO.output(trigger_gpio, False)
        timeout = time.time() + 0.04
        while GPIO.input(echo_gpio) == 0 and time.time() < timeout:
            pass
        pulse_start = time.time()
        while GPIO.input(echo_gpio) == 1 and time.time() < timeout:
            pass
        pulse_end = time.time()
        duration = pulse_end - pulse_start
        return round(duration * 17150, 2) if duration > 0 else -1
    except Exception:
        return -1


class UltrasonicSystem:
    """Ultrasonic sensor wrapper that supports gpiozero and manual fallback."""
    def __init__(self):
        self.use_manual = False
        self.sensor_front_left = None
        self.sensor_front_right = None
        self.sensor_back = None
        try:
            try:
                from gpiozero.pins.pigpio import PiGPIOFactory
                factory = PiGPIOFactory()
                self.sensor_front_left = DistanceSensor(echo=ECHO_FRONT_LEFT, trigger=TRIG_FRONT_LEFT, max_distance=4, pin_factory=factory)
                self.sensor_front_right = DistanceSensor(echo=ECHO_FRONT_RIGHT, trigger=TRIG_FRONT_RIGHT, max_distance=4, pin_factory=factory)
                self.sensor_back = DistanceSensor(echo=ECHO_BACK, trigger=TRIG_BACK, max_distance=4, pin_factory=factory)
            except Exception:
                self.sensor_front_left = DistanceSensor(echo=ECHO_FRONT_LEFT, trigger=TRIG_FRONT_LEFT, max_distance=4)
                self.sensor_front_right = DistanceSensor(echo=ECHO_FRONT_RIGHT, trigger=TRIG_FRONT_RIGHT, max_distance=4)
                self.sensor_back = DistanceSensor(echo=ECHO_BACK, trigger=TRIG_BACK, max_distance=4)
            self.enabled = True
        except Exception:
            self.use_manual = True
            self.enabled = True

    def get_distance(self, sensor_name: str):
        if not self.enabled:
            return -1
        try:
            if self.use_manual:
                if sensor_name == "front_left":
                    return _read_hcsr04_manual(TRIG_FRONT_LEFT, ECHO_FRONT_LEFT)
                if sensor_name == "front_right":
                    return _read_hcsr04_manual(TRIG_FRONT_RIGHT, ECHO_FRONT_RIGHT)
                if sensor_name == "back":
                    return _read_hcsr04_manual(TRIG_BACK, ECHO_BACK)
                if sensor_name == "front":
                    fl = _read_hcsr04_manual(TRIG_FRONT_LEFT, ECHO_FRONT_LEFT)
                    fr = _read_hcsr04_manual(TRIG_FRONT_RIGHT, ECHO_FRONT_RIGHT)
                    if fl < 0 and fr < 0:
                        return -1
                    if fl < 0:
                        return fr
                    if fr < 0:
                        return fl
                    return min(fl, fr)
                if sensor_name == "left":
                    return _read_hcsr04_manual(TRIG_FRONT_LEFT, ECHO_FRONT_LEFT)
                if sensor_name == "right":
                    return _read_hcsr04_manual(TRIG_FRONT_RIGHT, ECHO_FRONT_RIGHT)
                return -1

            if sensor_name == "front_left":
                d = self.sensor_front_left.distance * 100
            elif sensor_name == "front_right":
                d = self.sensor_front_right.distance * 100
            elif sensor_name == "back":
                d = self.sensor_back.distance * 100
            elif sensor_name == "front":
                fl = self.get_distance("front_left")
                fr = self.get_distance("front_right")
                if fl < 0 and fr < 0:
                    return -1
                if fl < 0:
                    return fr
                if fr < 0:
                    return fl
                return min(fl, fr)
            elif sensor_name == "left":
                return self.get_distance("front_left")
            elif sensor_name == "right":
                return self.get_distance("front_right")
            else:
                return -1
            if d is not None and d >= 0 and d <= 400:
                return round(d, 2)
            return -1
        except Exception:
            return -1

    def get_all_distances(self):
        fl = self.get_distance("front_left")
        fr = self.get_distance("front_right")
        back = self.get_distance("back")
        front = min(fl, fr) if fl >= 0 and fr >= 0 else (fl if fl >= 0 else fr)
        return {"front": front, "left": fl, "right": fr, "back": back}
