import time
from gpiozero import Motor
from .constants import LEFT_IN1, LEFT_IN2, LEFT_EN, RIGHT_IN3, RIGHT_IN4, RIGHT_EN
from .robot_state import get_critical_stop


class MotorController:
    """Motor control for L298N using gpiozero Motor objects."""
    def __init__(self):
        self.motor_left = Motor(forward=LEFT_IN1, backward=LEFT_IN2, enable=LEFT_EN)
        self.motor_right = Motor(forward=RIGHT_IN3, backward=RIGHT_IN4, enable=RIGHT_EN)
        self.speed = 0.85
        self.enabled = True
        self.stop()

    def set_speed(self, speed: float):
        self.speed = max(0.0, min(1.0, speed / 100.0))

    def _check_critical_and_stop(self):
        if get_critical_stop():
            self.stop()
            return True
        return False

    def forward(self):
        if not self.enabled:
            return
        if self._check_critical_and_stop():
            return
        self.motor_left.forward(self.speed)
        self.motor_right.forward(self.speed)

    def backward(self):
        if not self.enabled:
            return
        if self._check_critical_and_stop():
            return
        self.motor_left.backward(self.speed)
        self.motor_right.backward(self.speed)

    def left(self):
        if not self.enabled:
            return
        if self._check_critical_and_stop():
            return
        self.motor_left.backward(self.speed)
        self.motor_right.forward(self.speed)

    def right(self):
        if not self.enabled:
            return
        if self._check_critical_and_stop():
            return
        self.motor_left.forward(self.speed)
        self.motor_right.backward(self.speed)

    def stop(self):
        if not self.enabled:
            return
        self.motor_left.stop()
        self.motor_right.stop()

    def emergency_stop(self):
        self.stop()

    def forward_for(self, seconds: float):
        if not self.enabled:
            return
        self.forward()
        start = time.time()
        while time.time() - start < seconds:
            if get_critical_stop():
                self.stop()
                return
            time.sleep(0.02)
        self.stop()

    def backward_for(self, seconds: float):
        if not self.enabled:
            return
        self.backward()
        start = time.time()
        while time.time() - start < seconds:
            if get_critical_stop():
                self.stop()
                return
            time.sleep(0.02)
        self.stop()

    def pivot_left_for(self, seconds: float):
        if not self.enabled:
            return
        self.left()
        start = time.time()
        while time.time() - start < seconds:
            if get_critical_stop():
                self.stop()
                return
            time.sleep(0.02)
        self.stop()

    def pivot_right_for(self, seconds: float):
        if not self.enabled:
            return
        self.right()
        start = time.time()
        while time.time() - start < seconds:
            if get_critical_stop():
                self.stop()
                return
            time.sleep(0.02)
        self.stop()

    def cleanup(self):
        self.stop()
        if self.enabled:
            self.motor_left.close()
            self.motor_right.close()
