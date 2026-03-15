import os

# ======= Hardware Configuration =======
LEFT_IN1 = 17
LEFT_IN2 = 27
LEFT_EN = 24

RIGHT_IN3 = 22
RIGHT_IN4 = 23
RIGHT_EN = 25

TRIG_FRONT_LEFT = 5
ECHO_FRONT_LEFT = 6
TRIG_FRONT_RIGHT = 13
ECHO_FRONT_RIGHT = 19
TRIG_BACK = 20
ECHO_BACK = 21

CRITICAL_CM = 25.0
WARNING_CM = 45.0
SAFE_CM = 80.0
OBSTACLE_TURN_CM_MIN = 15.0
OBSTACLE_TURN_CM_MAX = 20.0
CAM_CLOSE_AREA = 25000
CAM_WARN_AREA = 12000

PI_CAM_WIDTH, PI_CAM_HEIGHT = 1280, 720

EXPLORE_NUM_SEGMENTS = 6
EXPLORE_MOVE_SEC_PER_SEGMENT = 2.0
EXPLORE_TURN_SEC = 0.9
EXPLORE_OBSTACLE_CM = 25.0
EXPLORE_DANGER_CM = 20.0
EXPLORE_CAUTION_MIN_CM = 15.0
EXPLORE_CAUTION_MAX_CM = 30.0
EXPLORE_BUBBLE_CM = 60.0

# Local resources and model directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")
VISION_DIR = os.path.join(SCRIPT_DIR, "..", "vision")
FACE_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "face_data")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VISION_DIR, exist_ok=True)
os.makedirs(FACE_DATA_DIR, exist_ok=True)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
CHAT_MODEL = "openai/gpt-3.5-turbo"
FALLBACK_CHAT_MODELS = ["openai/gpt-4o-mini", "openai/gpt-3.5-turbo", "openai/gpt-4o"]
