import threading
from enum import Enum

class RobotState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    MOVING = "moving"
    AVOIDING = "avoiding"
    EDGE_DETECTED = "edge_detected"
    EXPLORING = "exploring"
    STOPPED = "stopped"

critical_stop_flag = False
critical_stop_lock = threading.Lock()

obstacle_turn_lock = threading.Lock()
obstacle_turn_in_progress = False

robot_state = RobotState.IDLE
robot_state_lock = threading.Lock()


def set_critical_stop(flag: bool):
    global critical_stop_flag
    with critical_stop_lock:
        critical_stop_flag = flag


def get_critical_stop() -> bool:
    with critical_stop_lock:
        return critical_stop_flag


def set_robot_state(state: RobotState):
    global robot_state
    with robot_state_lock:
        robot_state = state


def get_robot_state() -> RobotState:
    with robot_state_lock:
        return robot_state
