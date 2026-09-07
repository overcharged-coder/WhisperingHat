import os

# Keep CPU thread usage reasonable on Raspberry Pi.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import csv
import json
import sys
import cv2
import time
import math
import subprocess
import shutil
import tempfile
import threading
import queue
import pickle
import numpy as np
from pathlib import Path
from contextlib import contextmanager

# Optional USB microphone voice control.
# Microphone capture uses SpeechRecognition/PyAudio, while speech-to-text uses
# Vosk entirely on-device (no Internet connection or cloud API required).
try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    from vosk import Model as VoskModel, KaldiRecognizer
except ImportError:
    VoskModel = None
    KaldiRecognizer = None
from collections import defaultdict, deque

SCRIPT_DIR = Path(__file__).resolve().parent

from ultralytics import YOLO

try:
    import face_recognition
except ImportError:
    face_recognition = None

# SparkFun Qwiic VL53L5CX driver. The program remains camera-only if missing.
try:
    import qwiic_vl53l5cx
except ImportError:
    qwiic_vl53l5cx = None

# LiteRT/TFLite interpreter fallback chain.
# On Raspberry Pi OS, use whichever package is available.
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
        except ImportError:
            Interpreter = None


# ============================================================
# WHISPERING HAT - PI 5 + CAMERA MODULE 3 WIDE + VL53L5CX TOF
# ============================================================
#
# What this version changes from your V4:
#   1. Adds Raspberry Pi Camera Module support through Picamera2.
#   2. Keeps USB webcam / video-file support through OpenCV.
#   3. Adds the walking corridor back as a trapezoid, not just a flat center box.
#   4. Makes objects inside the walking path more important than side objects.
#   5. Uses tracking so each detected object gets separate alert and distance memory.
#   6. Keeps calibrated fake distance and smoothing.
#   7. Adds optional reference-height distance, but leaves it OFF by default.
#   8. Keeps face identity and emotion inference on independent schedules.
#
# Important:
#   A single camera still cannot know true distance reliably.
#   For a wearable, use sonar / ToF later as the real distance source.
#
# ============================================================


# ============================================================
# SOURCE SETTINGS - RASPBERRY PI 5 + RASPBERRY PI OS TRIXIE
#                  + OFFICIAL PI CAMERA MODULE 3
# ============================================================

# For Pi 5 + Camera Module 3, use Picamera2/libcamera directly.
# Do not use the old PiCamera package or the legacy camera stack.
CAMERA_BACKEND = "picamera2"

# Camera Module 3 on Pi 5 is usually camera 0.
# If you attach two cameras, change this to 1 for the second camera.
PICAMERA2_CAMERA_INDEX = 0

# Used only if you change CAMERA_BACKEND to "opencv" for a USB webcam/video file.
OPENCV_SOURCE = 0

# Good starter resolution for Pi 5 CPU YOLO.
# 640x480 gives a stable 4:3 image and keeps CPU load reasonable.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 20

# Camera Module 3 autofocus.
# Options: "continuous", "auto", "manual", "off"
# For a wearable, continuous is usually best.
CAMERA3_AF_MODE = "continuous"

# Used only when CAMERA3_AF_MODE = "manual".
# LensPosition is in diopters: 0.0 is infinity, larger values focus closer.
# Try 1.0 to 3.0 for hallway/walking distance, 6.0+ for close objects.
CAMERA3_MANUAL_LENS_POSITION = 2.0

# Flip only if your camera is mounted upside down or mirrored.
CAMERA_HFLIP = False
CAMERA_VFLIP = True

# In SSH/headless mode, leave this False.
# On the Pi desktop or through VNC, change it to True to see the preview window.
SHOW_WINDOW = True

# Pi 5 performance settings.
# Start conservative. If it is smooth, try IMG_SIZE=416 or PROCESS_EVERY_N_FRAMES=2.
IMG_SIZE = 320
PROCESS_EVERY_N_FRAMES = 3

# Capture continuously in a background thread and always process the newest frame.
# This prevents YOLO from building up a queue of stale camera frames.
USE_LATEST_FRAME_THREAD = True
LATEST_FRAME_WAIT_SECONDS = 1.5

CONFIDENCE = 0.40
DEVICE = "cpu"

# Start with the normal PyTorch model.
# After exporting to NCNN, change this to "yolo11n_ncnn_model" for better Pi performance.
MODEL_NAME = "yolo11n.pt"

# Tracking keeps separate memory for different people and objects.
# It costs some extra CPU, but prevents objects in the same zone from sharing state.
USE_TRACKING = True
TRACKER_NAME = "bytetrack.yaml"

cv2.setNumThreads(1)

# Load the fallback Haar face detector once instead of rebuilding it repeatedly.
# Some Raspberry Pi OpenCV builds do not provide cv2.data, so try several
# common locations safely.
def find_haar_cascade_path():
    candidates = []

    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        candidates.append(
            Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        )

    candidates.extend([
        Path("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"),
        Path("/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"),
        Path("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"),
    ])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


HAAR_CASCADE_PATH = find_haar_cascade_path()

if HAAR_CASCADE_PATH is not None:
    HAAR_FACE_CASCADE = cv2.CascadeClassifier(str(HAAR_CASCADE_PATH))
    print(f"Loaded Haar face cascade: {HAAR_CASCADE_PATH}")
else:
    HAAR_FACE_CASCADE = cv2.CascadeClassifier()
    print("Haar face cascade not found. Haar fallback face detection is disabled.")


# ============================================================
# WALKING CORRIDOR SETTINGS
# ============================================================
#
# The corridor is a trapezoid/funnel because the path ahead appears narrower
# near the horizon and wider near the bottom of the image.
#
# Use the object's bottom-center point, not its box center, because the bottom
# of the object is closer to where it touches the ground.

USE_WALKING_CORRIDOR = True

# Top of the corridor as a fraction of image height.
# 0.30 means the corridor starts 30% down from the top.
CORRIDOR_TOP_Y = 0.32

# Width near the top/horizon.
CORRIDOR_TOP_LEFT = 0.43
CORRIDOR_TOP_RIGHT = 0.57

# Width near your feet / bottom of frame.
CORRIDOR_BOTTOM_LEFT = 0.18
CORRIDOR_BOTTOM_RIGHT = 0.82

# If an object is very low in the frame, it may still matter even near the side.
VERY_LOW_OBJECT_Y = 0.78

DRAW_CORRIDOR = True


# ============================================================
# DISTANCE SETTINGS
# ============================================================
#
# Best practical method for now:
#   calibrated_height
#
# Why not reference sizes by default?
#   Because YOLO box height depends on whether the whole object is visible,
#   camera angle, crop, lens distortion, pose, and object type.
#
# Calibration is easier:
#   1. Put a person/object at a known distance, like 1.2 m.
#   2. Read the on-screen h=___px label.
#   3. Update known_box_height_px below.

DISTANCE_MODE = "calibrated_height"
# Options:
#   "calibrated_height" = recommended
#   "reference_height"  = optional rough formula; needs focal length calibration
#   "area_fallback"     = roughest fallback

DISTANCE_CALIBRATION = {
    "person": {
        "known_distance_m": 1.2,
        "known_box_height_px": 280,
    },
    "chair": {
        "known_distance_m": 1.2,
        "known_box_height_px": 220,
    },
    "bench": {
        "known_distance_m": 1.2,
        "known_box_height_px": 220,
    },
    "backpack": {
        "known_distance_m": 1.0,
        "known_box_height_px": 180,
    },
    "handbag": {
        "known_distance_m": 1.0,
        "known_box_height_px": 160,
    },
    "suitcase": {
        "known_distance_m": 1.0,
        "known_box_height_px": 220,
    },
}

# Optional rough reference heights in meters.
# Keep this OFF unless you calibrate FOCAL_LENGTH_PX for your exact camera mode.
REFERENCE_HEIGHT_M = {
    "person": 1.68,      # rough U.S. adult average across male/female averages
    "chair": 0.90,      # very rough full chair height
    "bench": 0.80,      # very rough full bench height
    "backpack": 0.45,   # very variable
    "handbag": 0.35,    # very variable
    "suitcase": 0.65,   # carry-on-ish, very variable
    "dog": 0.45,        # very variable
    "cat": 0.25,        # very variable
}

# Focal length in pixels for reference-height distance.
# Better: set this by calibration using a full visible person/object.
# Example formula:
#   focal_px = box_height_px_at_known_distance * known_distance_m / real_height_m
FOCAL_LENGTH_PX = None

MIN_FAKE_DISTANCE_M = 0.25
MAX_FAKE_DISTANCE_M = 8.0

# Lower alpha = smoother but slower to react.
# Higher alpha = faster but jumpier.
DISTANCE_SMOOTH_ALPHA = 0.25

VERY_CLOSE_M = 0.65
CLOSE_M = 1.25
MEDIUM_M = 2.50

# For people, normal talking distance should not become endless STOP.
PERSON_STOP_DISTANCE_M = 0.55

APPROACHING_MPS = 0.25
FAST_APPROACHING_MPS = 0.70

AREA_GROWTH_APPROACHING = 0.22
AREA_GROWTH_FAST = 0.45


# ============================================================
# VL53L5CX QWIIC TOF SETTINGS
# ============================================================
#
# The ToF board can be connected and tested immediately. However, camera/ToF
# fusion is intentionally disabled until both boards are rigidly mounted.
# Once mounted together and pointing in the same direction, change:
#     TOF_MOUNT_CALIBRATED = True
# Then tune the overlap offsets while viewing the preview.

ENABLE_TOF = True
TOF_REQUIRED = False
TOF_MOUNT_CALIBRATED = True

# 8x8 gives 64 depth zones. The VL53L5CX supports at most 15 Hz in 8x8 mode.
TOF_RESOLUTION = 64
TOF_FREQUENCY_HZ = 12
TOF_RECONNECT_SECONDS = 3.0
TOF_STALE_SECONDS = 0.75
TOF_PRINT_SUMMARY = True
TOF_PRINT_INTERVAL_SECONDS = 1.0

# Camera Module 3 Wide is about 102 degrees horizontal and 67 degrees vertical.
# The VL53L5CX sensing area is about 45 x 45 degrees. With parallel optical
# axes, this is roughly the central 33.5% of image width and 62.5% of height.
# These are starting values only; actual Picamera2 sensor mode/cropping matters.
TOF_OVERLAP_WIDTH_FRACTION = 0.335
TOF_OVERLAP_HEIGHT_FRACTION = 0.625
TOF_OVERLAP_CENTER_X_OFFSET = 0.0
TOF_OVERLAP_CENTER_Y_OFFSET = 0.0

# Change these if the displayed depth grid is mirrored/upside down after mount.
# SparkFun's example reverses columns when printing, so X flip is a useful
# starting point, but the correct value depends on board orientation.
TOF_FLIP_X = True
TOF_FLIP_Y = False

# Use only plausible, recent measurements. Status 5 and 9 are commonly treated
# as usable VL53L5CX ranges. If the Python driver does not expose status, the
# distance is still range-checked.
TOF_VALID_STATUS_CODES = {5, 9}
TOF_MIN_DISTANCE_MM = 50
TOF_MAX_DISTANCE_MM = 4000
TOF_MIN_VALID_CELLS = 1
TOF_DISTANCE_PERCENTILE = 25.0

# Sample the middle/lower part of a YOLO box. This reduces background depth
# leaking into narrow object boxes while still catching a person's torso.
TOF_BOX_X_INSET_FRACTION = 0.18
TOF_BOX_Y_START_FRACTION = 0.22
TOF_CELL_PADDING = 0

# Unknown-obstacle warnings are powerful but should only be enabled after the
# ToF is rigidly mounted and its central path region is verified.
TOF_UNKNOWN_OBSTACLE_ENABLED = True
TOF_UNKNOWN_PATH_COL_START = 2
TOF_UNKNOWN_PATH_COL_END = 5
TOF_UNKNOWN_PATH_ROW_START = 2
TOF_UNKNOWN_PATH_ROW_END = 7
TOF_UNKNOWN_NOTICE_M = 2.5
TOF_UNKNOWN_CAUTION_M = 1.25
TOF_UNKNOWN_STOP_M = 0.55
TOF_UNKNOWN_MATCH_TOLERANCE_M = 0.45

# Time-to-collision thresholds used only with real ToF distance.
TOF_TTC_CAUTION_SECONDS = 2.0
TOF_TTC_STOP_SECONDS = 0.9

# ============================================================
# LOGGING SETTINGS
# ============================================================

LOG_CSV = True
LOG_FILE = str(SCRIPT_DIR / "whisper_hat_tof_log.csv")
LOG_ALL_DETECTIONS = False


# ============================================================
# SOUND SETTINGS
# ============================================================

# Options:
#   "off"      = no sound
#   "bell"     = terminal bell, may or may not make sound over SSH
#   "espeak"   = speak alert text if espeak is installed
#   "auto"     = Windows beep; on Linux terminal bell
ALERT_SOUND_MODE = "espeak"

# USB/PipeWire speech settings.
SPEECH_VOICE = "en-us"
SPEECH_RATE = 145
SPEAK_STARTUP_MESSAGE = True

# PipeWire normally lives in /run/user/<uid>. Explicitly passing this to
# pw-play fixes silent playback when Python is started from SSH, VNC, or an IDE.
AUDIO_RUNTIME_DIR = f"/run/user/{os.getuid()}"
PRINT_SPEECH_DEBUG = True

# USB microphone voice-control settings.
# Exact command phrases:
#   "detection off" = pause YOLO, face/emotion, and ToF hazard alerts
#   "detection on"  = resume detection
ENABLE_VOICE_CONTROL = True
VOICE_MIC_DEVICE_INDEX = 0      # Tested USB Microphone: Audio (hw:2,0)
VOICE_MIC_SAMPLE_RATE = 44100   # Hardware capture rate that passed your mic test.
VOICE_RECOGNITION_SAMPLE_RATE = 16000  # Vosk receives mono 16-bit PCM at 16 kHz.
VOICE_LISTEN_TIMEOUT = 1.0
VOICE_PHRASE_TIME_LIMIT = 4.0
VOICE_AMBIENT_CALIBRATION_SECONDS = 2.0
# Download/extract a small English Vosk model into this folder once. After that,
# recognition is fully offline. Example folder: vosk-model-small-en-us-0.15
VOICE_VOSK_MODEL_PATH = str(SCRIPT_DIR / "vosk-model-small-en-us-0.15")
VOICE_COMMAND_COOLDOWN_SECONDS = 1.0
VOICE_CONFIRM_COMMANDS = True
VOICE_PRINT_DEBUG = True


# ============================================================
# FACE RECOGNITION SETTINGS
# ============================================================

# Face recognition runs locally on the Pi.
# Put consented reference photos in:
#   known_faces/Alex/front.jpg
#   known_faces/Alex/left.jpg
#   known_faces/Mom/1.jpg
ENABLE_FACE_RECOGNITION = True

KNOWN_FACES_DIR = str(SCRIPT_DIR / "known_faces")
FACE_CACHE_FILE = str(SCRIPT_DIR / "known_faces_cache.pkl")

# Set True after adding/changing known face photos, run once, then set back to False.
REBUILD_FACE_CACHE = False

# On Raspberry Pi CPU, keep this slow/occasional.
# Your YOLO already runs every PROCESS_EVERY_N_FRAMES, and this runs even less often.
FACE_RECOGNITION_EVERY_N_FRAMES = 9

# Lower = stricter. 0.50 is a good safety-minded starting point.
FACE_MATCH_TOLERANCE = 0.50

# Use HOG on Raspberry Pi CPU. CNN is much heavier.
FACE_DETECTOR_MODEL = "hog"

# 1 is fastest. Higher can improve accuracy but slows the Pi.
FACE_NUM_JITTERS = 1

# Do not try face recognition on tiny/far-away person boxes.
FACE_MIN_PERSON_BOX_HEIGHT = 70

# Padding around the YOLO person box before searching for a face.
FACE_CROP_PAD_FRAC = 0.08


# ============================================================
# EMOTION RECOGNITION SETTINGS
# ============================================================

# Uses the bundled FER TFLite model. The interpreter reads its actual
# input shape automatically. The included model uses 64x64 grayscale faces.
ENABLE_EMOTION_RECOGNITION = True
EMOTION_MODEL_FILE = str(SCRIPT_DIR / "emotion_model.tflite")

# The included FER model was trained with pixel values scaled to [-1, 1].
# Other common choices are "zero_to_one" and "raw".
EMOTION_INPUT_NORMALIZATION = "minus_one_to_one"

# Standard FER2013 output order. Change this only if your model uses another order.
EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

# Emotion inference is relatively cheap, but do not run it on every camera frame.
EMOTION_EVERY_N_FRAMES = 6
EMOTION_MIN_CONFIDENCE = 0.25

# Useful when SHOW_WINDOW=False or the program is launched through SSH.
PRINT_EMOTION_TO_TERMINAL = True
EMOTION_PRINT_COOLDOWN_SECONDS = 1.5

# Speak stable facial-expression estimates as low-priority information.
# Safety alerts always receive a higher speech-queue priority.
SPEAK_DETECTED_EMOTIONS = True
EMOTION_SPEAK_MIN_CONFIDENCE = 0.45
EMOTION_SPEAK_STABLE_READINGS = 2
EMOTION_SPEAK_COOLDOWN_SECONDS = 15.0
EMOTION_SPEAK_CHANGE_GAP_SECONDS = 4.0
EMOTION_SPEAK_NEUTRAL = False
EMOTION_SPEAK_MAX_DISTANCE_M = 3.0

# Smooth several predictions so the label does not flicker.
EMOTION_HISTORY_LENGTH = 5


# ============================================================
# YOLO CLASS FILTER
# ============================================================
# COCO IDs:
# 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck,
# 9 traffic light, 11 stop sign, 13 bench, 15 cat, 16 dog,
# 17 horse, 19 cow, 24 backpack, 26 handbag, 28 suitcase, 56 chair

YOLO_CLASSES_TO_KEEP = [
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
    9,   # traffic light
    11,  # stop sign
    13,  # bench
    15,  # cat
    16,  # dog
    17,  # horse
    19,  # cow
    24,  # backpack
    26,  # handbag
    28,  # suitcase
    56,  # chair
]


# ============================================================
# ALERT / MEMORY SETTINGS
# ============================================================

# Speak risk 1 notices as well as risk 2 and 3 warnings.
# Set this back to 2 later if ordinary notices are too talkative.
MIN_SPOKEN_RISK = 1

# Short stabilization delays prevent one-frame false detections while remaining responsive.
STABLE_SECONDS_FOR_NOTICE = 0.30
STABLE_SECONDS_FOR_CAUTION = 0.25
STABLE_SECONDS_FOR_STOP = 0.10

# Balanced alert-history behavior:
# Speak immediately when a hazard first appears, speak again if the risk rises,
# and give occasional reminders while the hazard remains present.
REPEAT_ALERTS_WHILE_STILL_PRESENT = True
SPEAK_ON_SAME_RISK_REASON_CHANGE = True

# Reminder timing for an unchanged ongoing hazard.
# Notices are infrequent, cautions are more frequent, and STOP repeats fastest.
NOTICE_REPEAT_SECONDS = 20.0
CAUTION_REPEAT_SECONDS = 8.0
STOP_REPEAT_SECONDS = 4.0

GLOBAL_NOTICE_GAP_SECONDS = 2.0
GLOBAL_CAUTION_GAP_SECONDS = 1.2
GLOBAL_STOP_GAP_SECONDS = 0.5

# A class/zone must disappear this long before its alert history is forgotten.
ALERT_REARM_AFTER_ABSENT_SECONDS = 4.0

# Distance and tracking histories can expire sooner without forgetting that an
# alert was already spoken.
TRACK_STALE_SECONDS = 3.0

LOWER_FRAME_IMPORTANCE = 0.55

# Print the current best risk once per second while testing.
PRINT_ALERT_DEBUG = True
ALERT_DEBUG_INTERVAL_SECONDS = 1.0

# Alert memory uses a stable class-and-zone key. Distance/motion memory still
# uses tracking IDs, so a temporary ByteTrack ID change does not silence speech.
USE_TRACK_ID_FOR_ALERT_KEY = False


# ============================================================
# OBJECT CLASSES BY NAME
# ============================================================

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
PERSON_CLASSES = {"person"}
ANIMAL_CLASSES = {"dog", "cat", "horse", "cow"}
STATIC_OBSTACLE_CLASSES = {"bench", "chair", "backpack", "handbag", "suitcase"}
PASSIVE_SIGN_CLASSES = {"stop sign", "traffic light"}


# ============================================================
# STATE
# ============================================================

track_history = defaultdict(lambda: deque(maxlen=8))
track_memory = {}
distance_memory = {}
last_global_alert_time = 0
last_alert_debug_time = 0.0

# A single worker speaks queued messages. A priority queue lets STOP and
# Caution messages move ahead of informational emotion announcements.
speech_queue = queue.PriorityQueue(maxsize=8)
speech_worker_thread = None
speech_sequence = 0
speech_sequence_lock = threading.Lock()

known_face_names = []
known_face_encodings = []
face_memory = {}
emotion_recognizer = None
emotion_history = defaultdict(lambda: deque(maxlen=EMOTION_HISTORY_LENGTH))
last_emotion_print = {}
emotion_speech_memory = {}


# ToF worker/state. Access to the latest grid is protected by a lock.
tof_worker = None
tof_state_lock = threading.Lock()
tof_state = {
    "connected": False,
    "initialized": False,
    "grid_mm": None,
    "status_grid": None,
    "last_update": 0.0,
    "last_error": None,
    "frame_counter": 0,
}
last_tof_summary_time = 0.0

# Current audio process is tracked so an urgent STOP can interrupt speech.
current_audio_process = None
current_audio_lock = threading.Lock()

# Voice-control state. Detection starts ON and can be toggled while the
# camera/program keeps running. threading.Event is safe to share across threads.
detection_enabled_event = threading.Event()
detection_enabled_event.set()
voice_control_stop_event = threading.Event()
voice_control_thread = None
last_voice_command_time = 0.0


# ============================================================
# BASIC HELPERS
# ============================================================

def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def clamp(value, low, high):
    return max(low, min(high, value))


def center_of_box(x1, y1, x2, y2):
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def box_width(x1, x2):
    return max(1, x2 - x1)


def box_height(y1, y2):
    return max(1, y2 - y1)


def box_area(x1, y1, x2, y2):
    return max(1, box_width(x1, x2) * box_height(y1, y2))


def direction_word(cx, frame_width):
    if cx < frame_width * 0.33:
        return "left"
    elif cx > frame_width * 0.66:
        return "right"
    return "ahead"


# ============================================================
# WALKING CORRIDOR HELPERS
# ============================================================

def corridor_bounds_at_y(y, frame_width, frame_height):
    """
    Returns corridor left/right x at a given y.
    The corridor widens from top to bottom.
    """
    top_y_px = CORRIDOR_TOP_Y * frame_height
    bottom_y_px = frame_height

    if y <= top_y_px:
        t = 0.0
    else:
        t = (y - top_y_px) / max(1.0, bottom_y_px - top_y_px)
        t = clamp(t, 0.0, 1.0)

    left_frac = CORRIDOR_TOP_LEFT + t * (CORRIDOR_BOTTOM_LEFT - CORRIDOR_TOP_LEFT)
    right_frac = CORRIDOR_TOP_RIGHT + t * (CORRIDOR_BOTTOM_RIGHT - CORRIDOR_TOP_RIGHT)

    return left_frac * frame_width, right_frac * frame_width


def object_in_walking_corridor(cx, foot_y, frame_width, frame_height):
    if not USE_WALKING_CORRIDOR:
        return True

    left_x, right_x = corridor_bounds_at_y(foot_y, frame_width, frame_height)
    return left_x <= cx <= right_x and foot_y >= CORRIDOR_TOP_Y * frame_height


def corridor_position_word(cx, foot_y, frame_width, frame_height):
    if not USE_WALKING_CORRIDOR:
        return direction_word(cx, frame_width)

    left_x, right_x = corridor_bounds_at_y(foot_y, frame_width, frame_height)

    if left_x <= cx <= right_x and foot_y >= CORRIDOR_TOP_Y * frame_height:
        return "in path"
    if cx < left_x:
        return "left side"
    return "right side"


def draw_walking_corridor(frame):
    if not (USE_WALKING_CORRIDOR and DRAW_CORRIDOR):
        return

    frame_height, frame_width = frame.shape[:2]

    top_y = int(CORRIDOR_TOP_Y * frame_height)
    bottom_y = frame_height - 1

    top_left = (int(CORRIDOR_TOP_LEFT * frame_width), top_y)
    top_right = (int(CORRIDOR_TOP_RIGHT * frame_width), top_y)
    bottom_left = (int(CORRIDOR_BOTTOM_LEFT * frame_width), bottom_y)
    bottom_right = (int(CORRIDOR_BOTTOM_RIGHT * frame_width), bottom_y)

    # Subtle translucent overlay.
    overlay = frame.copy()
    pts = [bottom_left, top_left, top_right, bottom_right]
    cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], (40, 40, 40))
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    cv2.line(frame, bottom_left, top_left, (255, 255, 255), 2)
    cv2.line(frame, bottom_right, top_right, (255, 255, 255), 2)
    cv2.line(frame, top_left, top_right, (255, 255, 255), 1)

    cv2.putText(
        frame,
        "walking corridor",
        (top_left[0], max(20, top_y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )


def make_alert_key(track_id, class_name, corridor_position):
    # Zone-based keys are intentionally stable for speech timing.
    if USE_TRACK_ID_FOR_ALERT_KEY and track_id != -1:
        return f"id:{track_id}"
    return f"zone:{class_name}:{corridor_position}"


def make_distance_key(track_id, class_name, corridor_position):
    # Distance and motion estimates should stay separate for individual objects.
    if track_id != -1:
        return f"id:{track_id}"
    return f"zone:{class_name}:{corridor_position}"


# ============================================================
# VL53L5CX TOF WORKER + CAMERA FUSION HELPERS
# ============================================================

class ToFDepthWorker:
    """Continuously reads the newest 8x8 VL53L5CX depth frame."""

    def __init__(self):
        self.stop_event = threading.Event()
        self.thread = None
        self.sensor = None

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        self.thread = threading.Thread(
            target=self._run,
            name="vl53l5cx-depth-worker",
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        sensor = self.sensor
        if sensor is not None:
            try:
                sensor.stop_ranging()
            except Exception:
                pass
        if self.thread is not None:
            self.thread.join(timeout=3.0)

    def _set_state(self, **updates):
        with tof_state_lock:
            tof_state.update(updates)

    def _initialize_sensor(self):
        if qwiic_vl53l5cx is None:
            raise RuntimeError(
                "SparkFun ToF package missing. Install with: "
                "python -m pip install sparkfun-qwiic-vl53l5cx"
            )

        sensor = qwiic_vl53l5cx.QwiicVL53L5CX()
        if not sensor.is_connected():
            raise RuntimeError("VL53L5CX not found on I2C. Check Qwiic wiring and I2C.")

        print("[ToF] Initializing VL53L5CX; firmware upload can take several seconds...")
        if not sensor.begin():
            raise RuntimeError("VL53L5CX begin() failed")

        resolution = getattr(sensor, "kResolution8x8", TOF_RESOLUTION)
        status_ok = getattr(sensor, "kStatusOK", 0)

        if sensor.set_resolution(resolution) != status_ok:
            raise RuntimeError("VL53L5CX could not enter 8x8 mode")

        max_hz = 15 if int(TOF_RESOLUTION) == 64 else 60
        frequency = int(clamp(TOF_FREQUENCY_HZ, 1, max_hz))
        if sensor.set_ranging_frequency_hz(frequency) != status_ok:
            print(f"[ToF warning] Could not set frequency to {frequency} Hz; using sensor default.")

        target_closest = getattr(sensor, "kTargetOrderClosest", None)
        if target_closest is not None:
            try:
                sensor.set_target_order(target_closest)
            except Exception as error:
                print(f"[ToF warning] Could not set closest-target mode: {error}")

        if sensor.start_ranging() != status_ok:
            raise RuntimeError("VL53L5CX start_ranging() failed")

        self.sensor = sensor
        self._set_state(
            connected=True,
            initialized=True,
            last_error=None,
        )
        print(f"[ToF] Running 8x8 depth at requested {frequency} Hz.")

    @staticmethod
    def _reshape_optional(values):
        if values is None:
            return None
        array = np.asarray(values).reshape(-1)
        if array.size < 64:
            return None
        return array[:64].reshape(8, 8)

    def _store_measurement(self, measurement):
        distances = self._reshape_optional(getattr(measurement, "distance_mm", None))
        if distances is None:
            raise RuntimeError("ToF result did not contain 64 distance values")

        status_values = getattr(measurement, "status", None)
        if status_values is None:
            status_values = getattr(measurement, "target_status", None)
        status = self._reshape_optional(status_values)
        targets = self._reshape_optional(getattr(measurement, "nb_target_detected", None))

        distances = distances.astype(np.float32)
        valid = np.isfinite(distances)
        valid &= distances >= TOF_MIN_DISTANCE_MM
        valid &= distances <= TOF_MAX_DISTANCE_MM

        if status is not None:
            valid &= np.isin(status, list(TOF_VALID_STATUS_CODES))
        if targets is not None:
            valid &= targets > 0

        distances[~valid] = np.nan

        if TOF_FLIP_X:
            distances = np.fliplr(distances)
            if status is not None:
                status = np.fliplr(status)
        if TOF_FLIP_Y:
            distances = np.flipud(distances)
            if status is not None:
                status = np.flipud(status)

        with tof_state_lock:
            tof_state["connected"] = True
            tof_state["initialized"] = True
            tof_state["grid_mm"] = distances.copy()
            tof_state["status_grid"] = None if status is None else status.copy()
            tof_state["last_update"] = time.time()
            tof_state["last_error"] = None
            tof_state["frame_counter"] += 1

    def _run(self):
        while not self.stop_event.is_set():
            try:
                self._initialize_sensor()
                while not self.stop_event.is_set():
                    if self.sensor.check_data_ready():
                        measurement = self.sensor.get_ranging_data()
                        self._store_measurement(measurement)
                    else:
                        time.sleep(0.004)
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                print(f"[ToF error] {message}")
                self._set_state(
                    connected=False,
                    initialized=False,
                    last_error=message,
                )
                self.sensor = None
                # Missing software cannot recover by retrying. Wiring/I2C faults
                # may recover after reconnection, so those continue retrying.
                if TOF_REQUIRED or qwiic_vl53l5cx is None:
                    return
                self.stop_event.wait(TOF_RECONNECT_SECONDS)


def start_tof_worker():
    global tof_worker
    if not ENABLE_TOF:
        print("[ToF] Disabled in settings.")
        return
    if tof_worker is None:
        tof_worker = ToFDepthWorker()
    tof_worker.start()


def stop_tof_worker():
    if tof_worker is not None:
        tof_worker.stop()


def get_tof_snapshot():
    """Return a safe copy of the latest grid plus age/health information."""
    with tof_state_lock:
        grid = tof_state["grid_mm"]
        snapshot = {
            "connected": bool(tof_state["connected"]),
            "initialized": bool(tof_state["initialized"]),
            "grid_mm": None if grid is None else grid.copy(),
            "last_update": float(tof_state["last_update"]),
            "last_error": tof_state["last_error"],
            "frame_counter": int(tof_state["frame_counter"]),
        }
    snapshot["age_seconds"] = (
        float("inf")
        if snapshot["last_update"] <= 0
        else max(0.0, time.time() - snapshot["last_update"])
    )
    snapshot["fresh"] = (
        snapshot["grid_mm"] is not None
        and snapshot["age_seconds"] <= TOF_STALE_SECONDS
    )
    return snapshot


def maybe_print_tof_summary():
    global last_tof_summary_time
    if not (ENABLE_TOF and TOF_PRINT_SUMMARY):
        return
    now = time.time()
    if now - last_tof_summary_time < TOF_PRINT_INTERVAL_SECONDS:
        return
    last_tof_summary_time = now

    snapshot = get_tof_snapshot()
    grid = snapshot["grid_mm"]
    if snapshot["fresh"] and grid is not None and np.any(np.isfinite(grid)):
        valid = grid[np.isfinite(grid)]
        print(
            "[ToF] "
            f"closest={np.min(valid) / 1000.0:.2f}m "
            f"median={np.median(valid) / 1000.0:.2f}m "
            f"valid={valid.size}/64 "
            f"fusion={'ON' if TOF_MOUNT_CALIBRATED else 'LOCKED until mounted'}"
        )
    elif snapshot["last_error"]:
        print(f"[ToF] unavailable: {snapshot['last_error']}")
    else:
        print("[ToF] waiting for first depth frame...")


def tof_overlap_rectangle(frame_width, frame_height):
    width = TOF_OVERLAP_WIDTH_FRACTION * frame_width
    height = TOF_OVERLAP_HEIGHT_FRACTION * frame_height
    center_x = (0.5 + TOF_OVERLAP_CENTER_X_OFFSET) * frame_width
    center_y = (0.5 + TOF_OVERLAP_CENTER_Y_OFFSET) * frame_height

    left = clamp(center_x - width / 2, 0, frame_width - 1)
    right = clamp(center_x + width / 2, 1, frame_width)
    top = clamp(center_y - height / 2, 0, frame_height - 1)
    bottom = clamp(center_y + height / 2, 1, frame_height)
    return float(left), float(top), float(right), float(bottom)


def _pixel_to_tof_cell(x, y, overlap):
    left, top, right, bottom = overlap
    col = int((x - left) / max(1.0, right - left) * 8)
    row = int((y - top) / max(1.0, bottom - top) * 8)
    return int(clamp(col, 0, 7)), int(clamp(row, 0, 7))


def tof_distance_for_box(x1, y1, x2, y2, frame_width, frame_height):
    """Match a YOLO box to overlapping ToF cells.

    Returns None when the mount has not been calibrated, the box is outside the
    overlap, or no fresh valid cells are available.
    """
    if not (ENABLE_TOF and TOF_MOUNT_CALIBRATED):
        return None

    snapshot = get_tof_snapshot()
    if not snapshot["fresh"]:
        return None

    overlap = tof_overlap_rectangle(frame_width, frame_height)
    left, top, right, bottom = overlap

    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    sample_left = x1 + box_w * TOF_BOX_X_INSET_FRACTION
    sample_right = x2 - box_w * TOF_BOX_X_INSET_FRACTION
    sample_top = y1 + box_h * TOF_BOX_Y_START_FRACTION
    sample_bottom = y2

    inter_left = max(sample_left, left)
    inter_right = min(sample_right, right)
    inter_top = max(sample_top, top)
    inter_bottom = min(sample_bottom, bottom)

    if inter_right <= inter_left or inter_bottom <= inter_top:
        return None

    col0, row0 = _pixel_to_tof_cell(inter_left, inter_top, overlap)
    col1, row1 = _pixel_to_tof_cell(inter_right - 1e-3, inter_bottom - 1e-3, overlap)

    col0 = int(clamp(col0 - TOF_CELL_PADDING, 0, 7))
    col1 = int(clamp(col1 + TOF_CELL_PADDING, 0, 7))
    row0 = int(clamp(row0 - TOF_CELL_PADDING, 0, 7))
    row1 = int(clamp(row1 + TOF_CELL_PADDING, 0, 7))

    cells = snapshot["grid_mm"][row0:row1 + 1, col0:col1 + 1]
    valid = cells[np.isfinite(cells)]
    if valid.size < TOF_MIN_VALID_CELLS:
        return None

    distance_mm = float(np.percentile(valid, TOF_DISTANCE_PERCENTILE))
    return {
        "distance_m": distance_mm / 1000.0,
        "valid_cells": int(valid.size),
        "row_range": (row0, row1),
        "col_range": (col0, col1),
        "age_seconds": snapshot["age_seconds"],
        "overlap_rect": overlap,
    }


def tof_unknown_path_distance():
    if not (
        ENABLE_TOF
        and TOF_MOUNT_CALIBRATED
        and TOF_UNKNOWN_OBSTACLE_ENABLED
    ):
        return None
    snapshot = get_tof_snapshot()
    if not snapshot["fresh"]:
        return None
    grid = snapshot["grid_mm"]
    cells = grid[
        TOF_UNKNOWN_PATH_ROW_START:TOF_UNKNOWN_PATH_ROW_END + 1,
        TOF_UNKNOWN_PATH_COL_START:TOF_UNKNOWN_PATH_COL_END + 1,
    ]
    valid = cells[np.isfinite(cells)]
    if valid.size < TOF_MIN_VALID_CELLS:
        return None
    return {
        "distance_m": float(np.percentile(valid, TOF_DISTANCE_PERCENTILE)) / 1000.0,
        "valid_cells": int(valid.size),
        "age_seconds": snapshot["age_seconds"],
    }


def make_unknown_tof_debug(distance_m, valid_cells):
    return {
        "direction": "ahead",
        "corridor_position": "in path",
        "in_corridor": True,
        "lower_half": True,
        "distance_level": visual_distance_level(distance_m),
        "raw_distance_m": distance_m,
        "fake_distance_m": distance_m,
        "visual_distance_m": None,
        "distance_source": "tof_unknown",
        "approach_mps": 0.0,
        "approaching": False,
        "fast_approaching": False,
        "area_growth": 0.0,
        "speed_px": 0.0,
        "box_h_px": 0.0,
        "ttc_seconds": float("inf"),
        "tof_valid_cells": valid_cells,
        "tof_age_seconds": 0.0,
        "tof_row_range": None,
        "tof_col_range": None,
    }

# ============================================================
# DISTANCE ENGINE
# ============================================================

def visual_distance_level(distance_m):
    if distance_m < VERY_CLOSE_M:
        return "very_close"
    elif distance_m < CLOSE_M:
        return "close"
    elif distance_m < MEDIUM_M:
        return "medium"
    return "far"


def fallback_area_distance_meters(x1, y1, x2, y2, frame_width, frame_height):
    area = box_area(x1, y1, x2, y2)
    ratio = area / max(1, frame_width * frame_height)

    if ratio > 0.45:
        return 0.4
    elif ratio > 0.32:
        return 0.8
    elif ratio > 0.22:
        return 1.2
    elif ratio > 0.08:
        return 2.0
    elif ratio > 0.03:
        return 3.5
    return 5.0


def reference_height_distance_meters(class_name, x1, y1, x2, y2):
    h = box_height(y1, y2)

    if FOCAL_LENGTH_PX is None:
        return None

    if class_name not in REFERENCE_HEIGHT_M:
        return None

    return FOCAL_LENGTH_PX * REFERENCE_HEIGHT_M[class_name] / h


def calibrated_height_distance_meters(class_name, x1, y1, x2, y2):
    h = box_height(y1, y2)

    if class_name not in DISTANCE_CALIBRATION:
        return None

    cal = DISTANCE_CALIBRATION[class_name]
    distance_m = cal["known_distance_m"] * cal["known_box_height_px"] / h
    return distance_m


def raw_fake_distance_meters(class_name, x1, y1, x2, y2, frame_width, frame_height):
    if DISTANCE_MODE == "reference_height":
        d = reference_height_distance_meters(class_name, x1, y1, x2, y2)
        if d is not None:
            return clamp(d, MIN_FAKE_DISTANCE_M, MAX_FAKE_DISTANCE_M), "reference_height"

    if DISTANCE_MODE == "calibrated_height":
        d = calibrated_height_distance_meters(class_name, x1, y1, x2, y2)
        if d is not None:
            return clamp(d, MIN_FAKE_DISTANCE_M, MAX_FAKE_DISTANCE_M), "calibrated_height"

    d = fallback_area_distance_meters(x1, y1, x2, y2, frame_width, frame_height)
    return clamp(d, MIN_FAKE_DISTANCE_M, MAX_FAKE_DISTANCE_M), "area_fallback"


def update_distance_memory(distance_key, raw_distance_m, update=True):
    now = time.time()

    if distance_key not in distance_memory:
        distance_memory[distance_key] = {
            "raw_distance_m": raw_distance_m,
            "smooth_distance_m": raw_distance_m,
            "last_update_time": now,
            "approach_mps": 0.0,
        }
        return raw_distance_m, 0.0

    mem = distance_memory[distance_key]

    if not update:
        return mem["smooth_distance_m"], 0.0

    last_smooth = mem["smooth_distance_m"]
    last_time = mem["last_update_time"]
    dt = max(0.001, now - last_time)

    smooth = DISTANCE_SMOOTH_ALPHA * raw_distance_m + (1.0 - DISTANCE_SMOOTH_ALPHA) * last_smooth

    # Positive means distance is shrinking, so object is approaching.
    approach_mps = (last_smooth - smooth) / dt

    mem["raw_distance_m"] = raw_distance_m
    mem["smooth_distance_m"] = smooth
    mem["last_update_time"] = now
    mem["approach_mps"] = approach_mps

    return smooth, approach_mps


def estimate_motion(track_id):
    if track_id == -1:
        return 0, 0, 0, 0

    hist = track_history[track_id]
    if len(hist) < 2:
        return 0, 0, 0, 0

    old = hist[0]
    new = hist[-1]

    dx = new["cx"] - old["cx"]
    dy = new["cy"] - old["cy"]

    old_area = max(1, old["area"])
    new_area = max(1, new["area"])

    area_growth = (new_area - old_area) / old_area
    speed_px = math.sqrt(dx * dx + dy * dy)

    return dx, dy, area_growth, speed_px


# ============================================================
# CSV LOGGING
# ============================================================

CSV_FIELDS = [
    "time",
    "event",
    "class_name",
    "confidence",
    "risk",
    "reason",
    "direction",
    "corridor_position",
    "in_corridor",
    "distance_level",
    "distance_source",
    "raw_distance_m",
    "fake_distance_m",
    "visual_distance_m",
    "tof_valid_cells",
    "tof_age_seconds",
    "ttc_seconds",
    "approach_mps",
    "area_growth",
    "speed_px",
    "box_h_px",
    "face_name",
    "face_distance_score",
    "emotion",
    "emotion_confidence",
    "track_id",
]


def ensure_log_file():
    if not LOG_CSV:
        return

    path = Path(LOG_FILE)
    if path.exists():
        return

    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()


def log_row(event, class_name, confidence, risk, reason, debug, track_id):
    if not LOG_CSV:
        return

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow({
            "time": now_str(),
            "event": event,
            "class_name": class_name,
            "confidence": round(confidence, 3),
            "risk": risk,
            "reason": reason,
            "direction": debug["direction"],
            "corridor_position": debug["corridor_position"],
            "in_corridor": debug["in_corridor"],
            "distance_level": debug["distance_level"],
            "distance_source": debug["distance_source"],
            "raw_distance_m": round(debug["raw_distance_m"], 2),
            "fake_distance_m": round(debug["fake_distance_m"], 2),
            "visual_distance_m": (
                "" if debug.get("visual_distance_m") is None
                else round(debug.get("visual_distance_m"), 2)
            ),
            "tof_valid_cells": debug.get("tof_valid_cells", 0),
            "tof_age_seconds": (
                "" if debug.get("tof_age_seconds") is None
                else round(debug.get("tof_age_seconds"), 3)
            ),
            "ttc_seconds": (
                "" if not math.isfinite(debug.get("ttc_seconds", float("inf")))
                else round(debug.get("ttc_seconds"), 2)
            ),
            "approach_mps": round(debug["approach_mps"], 2),
            "area_growth": round(debug["area_growth"], 3),
            "speed_px": round(debug["speed_px"], 2),
            "box_h_px": round(debug["box_h_px"], 1),
            "face_name": debug.get("face_name", ""),
            "face_distance_score": (
                "" if debug.get("face_distance_score") is None
                else round(debug.get("face_distance_score"), 3)
            ),
            "emotion": debug.get("emotion", ""),
            "emotion_confidence": (
                "" if debug.get("emotion_confidence") is None
                else round(debug.get("emotion_confidence"), 3)
            ),
            "track_id": track_id,
        })


# ============================================================
# RISK ENGINE
# ============================================================

def apply_corridor_filter(class_name, risk, reason, distance_level, distance_m, approaching, fast_approaching, lower_half, in_corridor, corridor_position):
    """
    Makes the walking corridor matter.
    Objects in the path keep their normal risk.
    Side objects are downgraded unless they are very close or moving in.
    """
    if not USE_WALKING_CORRIDOR:
        return risk, reason

    if in_corridor:
        return risk, reason.replace("ahead", "in path") if "ahead" in reason else f"{reason} in path"

    # Static obstacles on the side usually should not trigger walking warnings.
    if class_name in STATIC_OBSTACLE_CLASSES:
        if lower_half and distance_level == "very_close":
            return min(risk, 1), f"side obstacle very close {corridor_position}"
        return 0, f"side {class_name} ignored"

    # People on the side are useful notice, but not usually STOP.
    if class_name in PERSON_CLASSES:
        if distance_m < PERSON_STOP_DISTANCE_M and fast_approaching:
            return 2, f"person moving in from {corridor_position}"
        if distance_level == "very_close":
            return min(risk, 2), f"person very close {corridor_position}"
        if distance_level == "close" and approaching:
            return min(risk, 2), f"person approaching from {corridor_position}"
        if distance_level in {"close", "medium"}:
            return min(risk, 1), f"person {corridor_position}"
        return 0, f"person far {corridor_position}"

    # Vehicles on the side can still matter if close or approaching.
    if class_name in VEHICLE_CLASSES:
        if distance_level in {"very_close", "close"} or fast_approaching:
            return min(max(risk, 2), 3), f"{class_name} close {corridor_position}"
        if approaching and distance_level == "medium":
            return min(max(risk, 2), 2), f"{class_name} approaching {corridor_position}"
        return min(risk, 1), f"{class_name} {corridor_position}"

    # Animals on side: warn only if close / approaching.
    if class_name in ANIMAL_CLASSES:
        if distance_level == "very_close" or (distance_level == "close" and approaching):
            return min(max(risk, 2), 2), f"{class_name} close {corridor_position}"
        if distance_level in {"close", "medium"}:
            return min(risk, 1), f"{class_name} {corridor_position}"
        return 0, f"{class_name} far {corridor_position}"

    # Signs and other objects outside path are passive.
    return min(risk, 1), f"{class_name} {corridor_position}"


def score_risk(class_name, cx, cy, x1, y1, x2, y2, frame_width, frame_height, track_id, distance_key, update_distance):
    direction = direction_word(cx, frame_width)
    foot_y = y2
    lower_half = cy > frame_height * LOWER_FRAME_IMPORTANCE
    very_low_object = foot_y > frame_height * VERY_LOW_OBJECT_Y

    in_corridor = object_in_walking_corridor(cx, foot_y, frame_width, frame_height)
    corridor_position = corridor_position_word(cx, foot_y, frame_width, frame_height)

    _, _, area_growth, speed_px = estimate_motion(track_id)

    visual_distance_m, visual_distance_source = raw_fake_distance_meters(
        class_name=class_name,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    tof_match = tof_distance_for_box(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    if tof_match is not None:
        raw_distance_m = tof_match["distance_m"]
        distance_source = "tof"
    else:
        raw_distance_m = visual_distance_m
        distance_source = visual_distance_source

    distance_m, approach_mps = update_distance_memory(
        distance_key=distance_key,
        raw_distance_m=raw_distance_m,
        update=update_distance,
    )

    distance_level = visual_distance_level(distance_m)

    approaching_by_distance = approach_mps > APPROACHING_MPS
    fast_approaching_by_distance = approach_mps > FAST_APPROACHING_MPS
    approaching_by_size = area_growth > AREA_GROWTH_APPROACHING
    fast_approaching_by_size = area_growth > AREA_GROWTH_FAST

    approaching = approaching_by_distance or approaching_by_size
    fast_approaching = fast_approaching_by_distance or fast_approaching_by_size

    if approach_mps > 0.10:
        ttc_seconds = distance_m / approach_mps
    else:
        ttc_seconds = float("inf")

    risk = 0
    reason = "passive"

    # ----------------------------
    # PERSON
    # ----------------------------
    if class_name in PERSON_CLASSES:
        if distance_m < PERSON_STOP_DISTANCE_M and fast_approaching:
            risk = 3
            reason = f"person rushing {direction}"
        elif distance_m < PERSON_STOP_DISTANCE_M:
            risk = 2
            reason = f"person very close {direction}"
        elif distance_level == "very_close" and approaching:
            risk = 2
            reason = f"person approaching close {direction}"
        elif distance_level == "very_close":
            risk = 2
            reason = f"person very close {direction}"
        elif distance_level == "close":
            risk = 2
            reason = f"person close {direction}"
        elif approaching and distance_level == "medium":
            risk = 2
            reason = f"person approaching {direction}"
        elif distance_level == "medium":
            risk = 1
            reason = f"person {direction}"
        else:
            risk = 0
            reason = f"person far {direction}"

    # ----------------------------
    # VEHICLES
    # ----------------------------
    elif class_name in VEHICLE_CLASSES:
        if distance_level == "very_close":
            risk = 3
            reason = f"{class_name} very close {direction}"
        elif fast_approaching and distance_level in {"close", "medium"}:
            risk = 3
            reason = f"{class_name} approaching fast {direction}"
        elif distance_level == "close":
            risk = 3
            reason = f"{class_name} close {direction}"
        elif approaching and distance_level == "medium":
            risk = 3
            reason = f"{class_name} approaching {direction}"
        elif distance_level == "medium":
            risk = 2
            reason = f"{class_name} {direction}"
        else:
            risk = 1
            reason = f"{class_name} nearby {direction}"

    # ----------------------------
    # ANIMALS
    # ----------------------------
    elif class_name in ANIMAL_CLASSES:
        if distance_level == "very_close" and approaching:
            risk = 3
            reason = f"{class_name} very close and approaching {direction}"
        elif distance_level == "very_close":
            risk = 2
            reason = f"{class_name} very close {direction}"
        elif distance_level == "close":
            risk = 2
            reason = f"{class_name} close {direction}"
        elif approaching and distance_level == "medium":
            risk = 2
            reason = f"{class_name} approaching {direction}"
        elif distance_level == "medium":
            risk = 1
            reason = f"{class_name} {direction}"
        else:
            risk = 0
            reason = f"{class_name} far {direction}"

    # ----------------------------
    # STATIC OBSTACLES
    # ----------------------------
    elif class_name in STATIC_OBSTACLE_CLASSES:
        path_relevant = lower_half or very_low_object
        if path_relevant and distance_level == "very_close":
            risk = 3
            reason = f"obstacle very close {direction}"
        elif path_relevant and distance_level == "close":
            risk = 2
            reason = f"obstacle close {direction}"
        elif path_relevant and approaching and distance_level == "medium":
            risk = 2
            reason = f"obstacle getting closer {direction}"
        elif path_relevant and distance_level == "medium":
            risk = 1
            reason = f"obstacle {direction}"
        elif distance_level == "very_close":
            risk = 1
            reason = f"{class_name} very close but not low {direction}"
        else:
            risk = 0
            reason = f"passive {class_name} {direction}"

    # ----------------------------
    # SIGNS / TRAFFIC LIGHTS
    # ----------------------------
    elif class_name in PASSIVE_SIGN_CLASSES:
        if distance_level == "very_close":
            risk = 1
            reason = f"{class_name} very close {direction}"
        else:
            risk = 0
            reason = f"passive {class_name} {direction}"

    # ----------------------------
    # OTHER OBJECTS
    # ----------------------------
    else:
        if lower_half and distance_level == "very_close":
            risk = 1
            reason = f"object close {direction}"
        else:
            risk = 0
            reason = f"passive {class_name} {direction}"

    risk, reason = apply_corridor_filter(
        class_name=class_name,
        risk=risk,
        reason=reason,
        distance_level=distance_level,
        distance_m=distance_m,
        approaching=approaching,
        fast_approaching=fast_approaching,
        lower_half=lower_half,
        in_corridor=in_corridor,
        corridor_position=corridor_position,
    )

    # Real depth enables time-to-collision decisions. This also catches the
    # wearer's own motion toward a stationary obstacle, which is useful for
    # collision avoidance.
    if distance_source == "tof" and in_corridor:
        if ttc_seconds <= TOF_TTC_STOP_SECONDS and distance_m <= CLOSE_M:
            risk = max(risk, 3)
            reason = f"{class_name} collision risk in path"
        elif ttc_seconds <= TOF_TTC_CAUTION_SECONDS and distance_m <= MEDIUM_M:
            risk = max(risk, 2)
            reason = f"{class_name} getting closer in path"

    debug = {
        "direction": direction,
        "corridor_position": corridor_position,
        "in_corridor": in_corridor,
        "lower_half": lower_half,
        "distance_level": distance_level,
        "raw_distance_m": raw_distance_m,
        "fake_distance_m": distance_m,
        "visual_distance_m": visual_distance_m,
        "distance_source": distance_source,
        "approach_mps": approach_mps,
        "ttc_seconds": ttc_seconds,
        "tof_valid_cells": 0 if tof_match is None else tof_match["valid_cells"],
        "tof_age_seconds": None if tof_match is None else tof_match["age_seconds"],
        "tof_row_range": None if tof_match is None else tof_match["row_range"],
        "tof_col_range": None if tof_match is None else tof_match["col_range"],
        "approaching": approaching,
        "fast_approaching": fast_approaching,
        "area_growth": area_growth,
        "speed_px": speed_px,
        "box_h_px": box_height(y1, y2),
    }

    return risk, reason, debug


# ============================================================
# OBJECT MEMORY + ALERT POLICY
# ============================================================

def update_object_memory(alert_key, risk, reason):
    """Update hazard state without forgetting previously spoken alerts.

    The alert is rearmed only when cleanup_stale_memory() confirms that the
    object/class-zone has actually been absent for several seconds.
    """
    now = time.time()

    if alert_key not in track_memory:
        track_memory[alert_key] = {
            "first_seen": now,
            "last_seen": now,
            "last_risk": risk,
            "last_reason": reason,
            "last_alert_time": 0.0,
            "last_alert_risk": 0,
            "last_alert_reason": None,
            "spoken_count": 0,
        }
        return track_memory[alert_key]

    mem = track_memory[alert_key]
    old_risk = mem.get("last_risk", 0)

    # Restart stabilization only when entering a spoken level for the first
    # time, or when the danger level escalates.
    if (
        risk >= MIN_SPOKEN_RISK
        and (old_risk < MIN_SPOKEN_RISK or risk > old_risk)
    ):
        mem["first_seen"] = now

    mem["last_seen"] = now
    mem["last_risk"] = risk
    mem["last_reason"] = reason

    return mem


def cleanup_stale_memory():
    now = time.time()

    # Forget spoken-alert history only after the hazard has truly disappeared.
    stale_alert_keys = [
        key
        for key, mem in track_memory.items()
        if now - mem["last_seen"] > ALERT_REARM_AFTER_ABSENT_SECONDS
    ]
    for key in stale_alert_keys:
        del track_memory[key]

    stale_distance_keys = [
        key
        for key, mem in distance_memory.items()
        if now - mem["last_update_time"] > TRACK_STALE_SECONDS
    ]
    for key in stale_distance_keys:
        del distance_memory[key]

    stale_history_ids = [
        track_id
        for track_id, history in track_history.items()
        if not history or now - history[-1]["time"] > TRACK_STALE_SECONDS
    ]
    for track_id in stale_history_ids:
        del track_history[track_id]


def alert_text(risk, reason):
    if risk == 3:
        return f"STOP. {reason}."
    elif risk == 2:
        return f"Caution. {reason}."
    elif risk == 1:
        return f"Notice. {reason}."
    return ""


def should_alert(alert_key, risk, reason):
    """Speak first alerts, escalations, meaningful changes, and timed reminders."""
    global last_global_alert_time

    if risk < MIN_SPOKEN_RISK:
        return False

    now = time.time()
    mem = track_memory.get(alert_key)
    if mem is None:
        return False

    stable_seconds = now - mem["first_seen"]

    if risk >= 3:
        stable_required = STABLE_SECONDS_FOR_STOP
        global_gap = GLOBAL_STOP_GAP_SECONDS
        repeat_seconds = STOP_REPEAT_SECONDS
    elif risk == 2:
        stable_required = STABLE_SECONDS_FOR_CAUTION
        global_gap = GLOBAL_CAUTION_GAP_SECONDS
        repeat_seconds = CAUTION_REPEAT_SECONDS
    else:
        stable_required = STABLE_SECONDS_FOR_NOTICE
        global_gap = GLOBAL_NOTICE_GAP_SECONDS
        repeat_seconds = NOTICE_REPEAT_SECONDS

    if stable_seconds < stable_required:
        return False

    never_spoken = mem["last_alert_time"] == 0
    risk_escalated = risk > mem["last_alert_risk"]
    reason_changed = reason != mem.get("last_alert_reason")
    repeat_due = (
        REPEAT_ALERTS_WHILE_STILL_PRESENT
        and mem["last_alert_time"] > 0
        and now - mem["last_alert_time"] >= repeat_seconds
    )
    global_gap_passed = now - last_global_alert_time >= global_gap

    should_speak = never_spoken or risk_escalated or repeat_due

    if SPEAK_ON_SAME_RISK_REASON_CHANGE and reason_changed:
        should_speak = True

    if not global_gap_passed and not risk_escalated:
        return False

    if not should_speak:
        return False

    mem["last_alert_time"] = now
    mem["last_alert_risk"] = risk
    mem["last_alert_reason"] = reason
    mem["spoken_count"] = mem.get("spoken_count", 0) + 1
    last_global_alert_time = now
    return True



def print_alert_debug(risk, reason, debug):
    global last_alert_debug_time

    if not PRINT_ALERT_DEBUG or debug is None:
        return

    now = time.time()
    if now - last_alert_debug_time < ALERT_DEBUG_INTERVAL_SECONDS:
        return

    last_alert_debug_time = now
    print(
        "[alert debug] "
        f"risk={risk} "
        f"distance={debug['fake_distance_m']:.2f}m "
        f"source={debug.get('distance_source')} "
        f"ttc={debug.get('ttc_seconds', float('inf')):.2f}s "
        f"position={debug['corridor_position']} "
        f"reason={reason}"
    )


def _speech_environment():
    """Return an environment that can connect to the user's PipeWire session."""
    env = os.environ.copy()
    env["XDG_RUNTIME_DIR"] = AUDIO_RUNTIME_DIR
    env.setdefault(
        "DBUS_SESSION_BUS_ADDRESS",
        f"unix:path={AUDIO_RUNTIME_DIR}/bus",
    )
    return env


def _speak_message_now(message):
    """Generate a WAV and play it; STOP can terminate the active player."""
    global current_audio_process

    speech_program = shutil.which("espeak-ng") or shutil.which("espeak")
    audio_player = shutil.which("pw-play")

    if speech_program is None:
        print("[speech error] espeak-ng/espeak was not found.")
        return
    if audio_player is None:
        print("[speech error] pw-play was not found.")
        return

    wav_path = None
    try:
        if PRINT_SPEECH_DEBUG:
            print(f"[speech start] {message}")

        with tempfile.NamedTemporaryFile(
            prefix="whisper_hat_", suffix=".wav", delete=False
        ) as temp_file:
            wav_path = temp_file.name

        speech_result = subprocess.run(
            [
                speech_program,
                "-v", SPEECH_VOICE,
                "-s", str(SPEECH_RATE),
                "-w", wav_path,
                message,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
        if speech_result.returncode != 0:
            error_text = speech_result.stderr.strip() or "unknown eSpeak error"
            print(f"[speech error] WAV generation failed: {error_text}")
            return

        process = subprocess.Popen(
            [audio_player, "--target=auto", wav_path],
            env=_speech_environment(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        with current_audio_lock:
            current_audio_process = process

        try:
            _stdout, stderr = process.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
            print("[speech error] speech playback timed out.")
            return

        # A STOP interruption terminates pw-play, so a nonzero code is expected.
        if process.returncode not in (0, -15):
            error_text = (stderr or "").strip() or f"pw-play code {process.returncode}"
            print(f"[speech warning] {error_text}")

        if PRINT_SPEECH_DEBUG:
            print("[speech complete]")

    except subprocess.TimeoutExpired:
        print("[speech error] speech generation timed out.")
    except Exception as error:
        print(f"[speech error] {type(error).__name__}: {error}")
    finally:
        with current_audio_lock:
            current_audio_process = None
        if wav_path:
            try:
                os.remove(wav_path)
            except OSError:
                pass


def _speech_worker():
    """Speak one queued message at a time, highest-risk messages first."""
    while True:
        item = speech_queue.get()

        try:
            _priority, _sequence, message = item
            if message is None:
                return

            _speak_message_now(message)
        finally:
            speech_queue.task_done()


def start_speech_worker():
    global speech_worker_thread

    if (
        speech_worker_thread is not None
        and speech_worker_thread.is_alive()
    ):
        return

    speech_worker_thread = threading.Thread(
        target=_speech_worker,
        name="whisper-hat-speech-worker",
        daemon=True,
    )
    speech_worker_thread.start()

    if PRINT_SPEECH_DEBUG:
        print(
            "[speech worker started] "
            f"runtime={AUDIO_RUNTIME_DIR}"
        )


def stop_speech_worker():
    """Allow the current queued message to finish when the program closes."""
    global speech_sequence

    if speech_worker_thread is None:
        return

    try:
        with speech_sequence_lock:
            speech_sequence += 1
            sequence = speech_sequence
        speech_queue.put_nowait((999, sequence, None))
    except queue.Full:
        pass

    speech_worker_thread.join(timeout=3.0)


def clear_waiting_speech():
    """Drop messages that have not begun playing."""
    while True:
        try:
            speech_queue.get_nowait()
            speech_queue.task_done()
        except queue.Empty:
            return


def interrupt_current_speech():
    """Immediately stop active pw-play so a STOP warning can take over."""
    with current_audio_lock:
        process = current_audio_process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass


def play_alert_sound(risk, message):
    """Queue speech without blocking camera and YOLO processing.

    Risk 3 receives the highest priority. Emotion announcements use risk 0,
    so they wait behind STOP, Caution, and Notice messages already queued.
    """
    global speech_sequence

    if ALERT_SOUND_MODE == "off" or not message:
        return False

    if ALERT_SOUND_MODE != "espeak":
        print("", end="", flush=True)
        return True

    start_speech_worker()

    # Risk 3 must not wait behind a face name, emotion, notice, or caution.
    if risk >= 3:
        clear_waiting_speech()
        interrupt_current_speech()

    try:
        with speech_sequence_lock:
            speech_sequence += 1
            sequence = speech_sequence

        priority = -int(risk)
        speech_queue.put_nowait((priority, sequence, message))
        return True
    except queue.Full:
        print("[speech warning] queue full, dropping this message.")
        return False

# ============================================================
# USB MICROPHONE VOICE CONTROL
# ============================================================

@contextmanager
def suppress_audio_stderr():
    """Temporarily silence native ALSA/JACK probing messages.

    PyAudio can print many harmless lines such as "Unknown PCM surround51"
    directly to file descriptor 2.  We use this only while constructing/opening
    the microphone, so normal Python/camera/ToF errors remain visible.
    """
    try:
        sys.stderr.flush()
    except Exception:
        pass

    stderr_fd = 2
    saved_stderr_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)


@contextmanager
def open_microphone_quietly(microphone):
    """Open/close a SpeechRecognition microphone without ALSA/JACK spam."""
    source = None
    with suppress_audio_stderr():
        source = microphone.__enter__()
    try:
        yield source
    finally:
        with suppress_audio_stderr():
            microphone.__exit__(None, None, None)


def detection_is_enabled():
    return detection_enabled_event.is_set()


def set_detection_enabled(enabled, announce=True):
    """Turn all hazard/object detection processing on or off."""
    changed = detection_is_enabled() != bool(enabled)

    if enabled:
        detection_enabled_event.set()
    else:
        detection_enabled_event.clear()
        # Do not allow old alerts to keep speaking after the user says OFF.
        clear_waiting_speech()
        interrupt_current_speech()
        track_memory.clear()
        distance_memory.clear()
        track_history.clear()
        face_memory.clear()
        emotion_history.clear()
        emotion_speech_memory.clear()

    if changed:
        state = "ON" if enabled else "OFF"
        print(f"[voice control] Detection {state}.")
        if announce and VOICE_CONFIRM_COMMANDS:
            play_alert_sound(2 if enabled else 1, f"Detection {state.lower()}.")

    return changed


def _select_voice_microphone_index():
    """Return the configured USB microphone index without scanning all ALSA devices."""
    if sr is None:
        return None

    if VOICE_MIC_DEVICE_INDEX is not None:
        index = int(VOICE_MIC_DEVICE_INDEX)
        print(f"[voice control] Using configured USB microphone device {index}.", flush=True)
        return index

    print("[voice control] No microphone index configured; using default input.")
    return None


def _normalize_voice_command(text):
    return " ".join(text.lower().strip().split())


def _apply_voice_command(text):
    """Handle only the two deliberate phrases: detection on / detection off."""
    global last_voice_command_time

    command = _normalize_voice_command(text)
    if command not in {"detection on", "detection off"}:
        return False

    now = time.time()
    if now - last_voice_command_time < VOICE_COMMAND_COOLDOWN_SECONDS:
        return True
    last_voice_command_time = now

    if command == "detection on":
        set_detection_enabled(True, announce=True)
    else:
        set_detection_enabled(False, announce=True)
    return True


def _voice_control_worker():
    """Listen for detection on/off using fully offline Vosk recognition."""
    if sr is None:
        print(
            "[voice control] SpeechRecognition is not installed. "
            "Install it with: pip install SpeechRecognition"
        )
        return

    if VoskModel is None or KaldiRecognizer is None:
        print(
            "[voice control] Vosk is not installed. "
            "Install it with: pip install vosk"
        )
        return

    model_path = Path(VOICE_VOSK_MODEL_PATH)
    if not model_path.exists():
        print(f"[voice control] Vosk model folder not found: {model_path}")
        print(
            "[voice control] Download a small English Vosk model once, extract it "
            "beside this script, then voice control works completely offline."
        )
        return

    try:
        vosk_model = VoskModel(str(model_path))
    except Exception as error:
        print(f"[voice control] Could not load Vosk model: {error}")
        return

    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    mic_index = _select_voice_microphone_index()

    try:
        # Creating a PyAudio-backed Microphone can probe many nonexistent ALSA
        # profiles. Silence only that native probing noise.
        with suppress_audio_stderr():
            microphone = sr.Microphone(
                device_index=mic_index,
                sample_rate=VOICE_MIC_SAMPLE_RATE,
            )
    except Exception as error:
        print(f"[voice control] Could not open microphone: {error}")
        print(
            "[voice control] On Raspberry Pi, install microphone support with: "
            "sudo apt install -y portaudio19-dev python3-pyaudio flac"
        )
        return

    try:
        with open_microphone_quietly(microphone) as source:
            print("[voice control] Calibrating USB microphone for ambient noise...", flush=True)
            recognizer.adjust_for_ambient_noise(
                source,
                duration=VOICE_AMBIENT_CALIBRATION_SECONDS,
            )
            if VOICE_PRINT_DEBUG:
                print(f"[voice control] Energy threshold: {recognizer.energy_threshold:.1f}")
    except Exception as error:
        print(f"[voice control] Microphone calibration warning: {error}")

    # Restrict decoding to exactly the phrases this project needs. This makes
    # a small offline model both faster and less prone to accidental commands.
    command_grammar = json.dumps(["detection on", "detection off", "[unk]"])

    print(
        '[voice control] OFFLINE Vosk ready. Say "detection off" or "detection on".',
        flush=True,
    )

    while not voice_control_stop_event.is_set():
        try:
            with open_microphone_quietly(microphone) as source:
                audio = recognizer.listen(
                    source,
                    timeout=VOICE_LISTEN_TIMEOUT,
                    phrase_time_limit=VOICE_PHRASE_TIME_LIMIT,
                )

            # Convert the captured audio locally to the PCM format Vosk expects.
            raw_audio = audio.get_raw_data(
                convert_rate=VOICE_RECOGNITION_SAMPLE_RATE,
                convert_width=2,
            )

            offline_recognizer = KaldiRecognizer(
                vosk_model,
                VOICE_RECOGNITION_SAMPLE_RATE,
                command_grammar,
            )
            offline_recognizer.AcceptWaveform(raw_audio)
            result = json.loads(offline_recognizer.FinalResult())
            text = _normalize_voice_command(result.get("text", ""))

            if VOICE_PRINT_DEBUG and text:
                print(f"[voice heard offline] {text}")

            if text:
                _apply_voice_command(text)

        except sr.WaitTimeoutError:
            continue
        except Exception as error:
            print(f"[voice control] Offline microphone/recognition error: {error}")
            voice_control_stop_event.wait(1.0)


def _voice_control_worker_safe():
    """Run voice control and make any unexpected thread failure visible."""
    try:
        _voice_control_worker()
    except Exception as error:
        print(
            f"[voice control] Worker crashed: {type(error).__name__}: {error}",
            flush=True,
        )


def start_voice_control():
    global voice_control_thread

    if not ENABLE_VOICE_CONTROL:
        print("[voice control] Disabled in settings.", flush=True)
        return

    if voice_control_thread is not None and voice_control_thread.is_alive():
        print("[voice control] Worker is already running.", flush=True)
        return

    # Print from the MAIN thread before launching the worker. This guarantees
    # visible confirmation that the voice startup path was reached.
    print(
        f"[voice control] Starting USB voice control on device "
        f"{VOICE_MIC_DEVICE_INDEX} at {VOICE_MIC_SAMPLE_RATE} Hz...",
        flush=True,
    )

    voice_control_stop_event.clear()
    voice_control_thread = threading.Thread(
        target=_voice_control_worker_safe,
        name="whisper-hat-voice-control",
        daemon=True,
    )
    voice_control_thread.start()


def stop_voice_control():
    voice_control_stop_event.set()
    if voice_control_thread is not None and voice_control_thread.is_alive():
        voice_control_thread.join(timeout=3.0)


# ============================================================
# DRAWING
# ============================================================

def draw_hud(frame, best_risk, best_reason, fps, detection_ran):
    frame_height, frame_width = frame.shape[:2]

    if best_risk == 3:
        text = f"STOP: {best_reason}"
        color = (0, 0, 255)
    elif best_risk == 2:
        text = f"CAUTION: {best_reason}"
        color = (0, 165, 255)
    elif best_risk == 1:
        text = f"NOTICE: {best_reason}"
        color = (0, 255, 255)
    else:
        text = "SAFE / passive"
        color = (0, 255, 0)

    cv2.rectangle(frame, (10, 35), (min(frame_width - 10, 900), 120), (0, 0, 0), -1)

    cv2.putText(frame, text, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2)

    mode = "YOLO" if detection_ran else "reuse"
    tracking_text = "track" if USE_TRACKING else "predict"

    cv2.putText(
        frame,
        f"FPS: {fps:.1f} | {tracking_text}/{mode} | imgsz={IMG_SIZE} | every {PROCESS_EVERY_N_FRAMES} frames",
        (20, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
    )


def draw_object_box(frame, x1, y1, x2, y2, class_name, conf, risk, track_id, debug):
    if risk == 3:
        color = (0, 0, 255)
    elif risk == 2:
        color = (0, 165, 255)
    elif risk == 1:
        color = (0, 255, 255)
    else:
        color = (0, 255, 0)

    x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), color, 2)

    corridor_tag = "PATH" if debug["in_corridor"] else "SIDE"

    display_name = class_name
    if class_name == "person" and debug.get("face_name"):
        display_name = debug["face_name"]

    if class_name == "person" and debug.get("emotion"):
        display_name += f" [{debug['emotion']} {debug.get('emotion_confidence', 0):.0%}]"

    label = (
        f"{display_name} {conf:.2f} risk={risk} {corridor_tag} "
        f"{debug['fake_distance_m']:.1f}m h={debug.get('box_h_px', 0):.0f}px"
    )

    if track_id != -1:
        label += f" id={track_id}"

    label_y = y1_i - 10
    if label_y < 135:
        label_y = y2_i + 22
    label_y = min(frame.shape[0] - 35, max(25, label_y))

    cv2.putText(frame, label, (x1_i, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)

    debug_text = (
        f"{debug['corridor_position']} | {debug['distance_level']} | "
        f"src={debug['distance_source']} | approach={debug['approach_mps']:.2f}m/s"
    )

    debug_y = min(frame.shape[0] - 10, label_y + 20)
    cv2.putText(frame, debug_text, (x1_i, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)




def draw_tof_overlay(frame):
    """Draw approximate camera/ToF overlap and the latest depth grid."""
    if not ENABLE_TOF:
        return
    frame_height, frame_width = frame.shape[:2]
    left, top, right, bottom = tof_overlap_rectangle(frame_width, frame_height)
    color = (255, 200, 0) if TOF_MOUNT_CALIBRATED else (150, 150, 150)
    cv2.rectangle(
        frame,
        (int(left), int(top)),
        (int(right), int(bottom)),
        color,
        1,
    )

    snapshot = get_tof_snapshot()
    label = "ToF calibrated" if TOF_MOUNT_CALIBRATED else "ToF overlap estimate - fusion locked"
    if snapshot["fresh"]:
        label += f" | age {snapshot['age_seconds']:.2f}s"
    cv2.putText(
        frame,
        label,
        (int(left), max(145, int(top) - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        color,
        1,
    )

    grid = snapshot["grid_mm"]
    if not snapshot["fresh"] or grid is None:
        return

    cell_w = (right - left) / 8.0
    cell_h = (bottom - top) / 8.0
    for row in range(8):
        for col in range(8):
            x0 = int(left + col * cell_w)
            y0 = int(top + row * cell_h)
            x1 = int(left + (col + 1) * cell_w)
            y1 = int(top + (row + 1) * cell_h)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 1)
            value = grid[row, col]
            if np.isfinite(value):
                cv2.putText(
                    frame,
                    f"{value / 1000.0:.1f}",
                    (x0 + 2, min(y1 - 2, y0 + 13)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.27,
                    color,
                    1,
                )


# ============================================================
# EMOTION RECOGNITION
# ============================================================

class EmotionRecognizer:
    def __init__(self, model_path):
        self.enabled = False
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_height = 48
        self.input_width = 48
        self.input_channels = 1

        if not ENABLE_EMOTION_RECOGNITION:
            print("Emotion recognition disabled.")
            return

        if Interpreter is None:
            print("No LiteRT/TFLite interpreter is installed. Emotion recognition disabled.")
            print("Try one of these inside your virtual environment:")
            print("  pip install ai-edge-litert")
            print("  pip install tflite-runtime")
            return

        model_path = Path(model_path)
        if not model_path.exists():
            print(f"Emotion model not found: {model_path}")
            print("Place emotion_model.tflite beside this script.")
            return

        try:
            self.interpreter = Interpreter(model_path=str(model_path), num_threads=2)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()[0]
            self.output_details = self.interpreter.get_output_details()[0]

            shape = self.input_details["shape"]
            # Expected NHWC, usually [1, 48, 48, 1].
            if len(shape) == 4:
                self.input_height = int(shape[1])
                self.input_width = int(shape[2])
                self.input_channels = int(shape[3])

            self.enabled = True
            print(
                "Emotion model loaded: "
                f"{model_path} input={tuple(shape)} "
                f"dtype={self.input_details['dtype']} "
                f"output={tuple(self.output_details['shape'])}"
            )
        except Exception as e:
            print(f"Could not load emotion model: {e}")
            self.enabled = False

    @staticmethod
    def _softmax(values):
        values = np.asarray(values, dtype=np.float32)
        values = values - np.max(values)
        exp = np.exp(values)
        total = np.sum(exp)
        return exp / total if total > 0 else exp

    def predict(self, face_bgr):
        if not self.enabled or face_bgr is None or face_bgr.size == 0:
            return None, None

        try:
            if self.input_channels == 1:
                image = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(
                    image,
                    (self.input_width, self.input_height),
                    interpolation=cv2.INTER_AREA,
                )
                image = image[..., np.newaxis]
            else:
                image = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                image = cv2.resize(
                    image,
                    (self.input_width, self.input_height),
                    interpolation=cv2.INTER_AREA,
                )

            input_dtype = self.input_details["dtype"]
            scale, zero_point = self.input_details.get("quantization", (0.0, 0))

            float_image = image.astype(np.float32)

            if EMOTION_INPUT_NORMALIZATION == "minus_one_to_one":
                float_image = (float_image / 255.0 - 0.5) * 2.0
            elif EMOTION_INPUT_NORMALIZATION == "zero_to_one":
                float_image = float_image / 255.0
            elif EMOTION_INPUT_NORMALIZATION == "raw":
                pass
            else:
                raise ValueError(
                    "Unknown EMOTION_INPUT_NORMALIZATION: "
                    f"{EMOTION_INPUT_NORMALIZATION}"
                )

            if np.issubdtype(input_dtype, np.floating):
                tensor = float_image.astype(input_dtype)
            else:
                # Quantized model: map the normalized floating-point values
                # into the tensor's quantized integer range.
                if scale and scale > 0:
                    tensor = np.round(float_image / scale + zero_point)
                else:
                    tensor = image
                limits = np.iinfo(input_dtype)
                tensor = np.clip(tensor, limits.min, limits.max).astype(input_dtype)

            tensor = np.expand_dims(tensor, axis=0)
            self.interpreter.set_tensor(self.input_details["index"], tensor)
            self.interpreter.invoke()

            output = self.interpreter.get_tensor(self.output_details["index"])[0]
            output = np.asarray(output, dtype=np.float32).reshape(-1)

            out_scale, out_zero = self.output_details.get("quantization", (0.0, 0))
            if not np.issubdtype(self.output_details["dtype"], np.floating) and out_scale:
                output = (output - out_zero) * out_scale

            # Some models already output probabilities; others output logits.
            if (
                np.any(output < 0)
                or np.any(output > 1)
                or not np.isclose(float(np.sum(output)), 1.0, atol=0.15)
            ):
                probabilities = self._softmax(output)
            else:
                probabilities = output / max(float(np.sum(output)), 1e-8)

            best_index = int(np.argmax(probabilities))
            if best_index >= len(EMOTION_LABELS):
                return None, None

            confidence = float(probabilities[best_index])
            emotion = EMOTION_LABELS[best_index]

            if confidence < EMOTION_MIN_CONFIDENCE:
                return "uncertain", confidence

            return emotion, confidence

        except Exception as e:
            print(f"Emotion inference failed: {e}")
            return None, None


def smooth_emotion(face_key, emotion, confidence):
    if emotion is None or confidence is None:
        return None, None

    emotion_history[face_key].append((emotion, confidence))

    totals = defaultdict(float)
    counts = defaultdict(int)

    for label, score in emotion_history[face_key]:
        totals[label] += float(score)
        counts[label] += 1

    best_label = max(totals, key=totals.get)
    average_confidence = totals[best_label] / max(1, counts[best_label])
    return best_label, average_confidence


# ============================================================
# FACE RECOGNITION HELPERS
# ============================================================

def face_name_from_path(image_path, root_dir):
    """
    Supports either:
      known_faces/Alex.jpg
      known_faces/Alex/1.jpg
      known_faces/Alex/2.jpg
    """
    root_dir = Path(root_dir)

    if image_path.parent == root_dir:
        return image_path.stem

    return image_path.parent.name


def load_known_faces():
    if not ENABLE_FACE_RECOGNITION:
        print("Face recognition disabled.")
        return [], []

    if face_recognition is None:
        print("face_recognition is not installed. Face recognition disabled.")
        print("Install later with:")
        print("  python3 -m pip install face_recognition")
        return [], []

    cache_path = Path(FACE_CACHE_FILE)

    if cache_path.exists() and not REBUILD_FACE_CACHE:
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)

            print(f"Loaded {len(data['names'])} known face encoding(s) from cache.")
            return data["names"], data["encodings"]
        except Exception as e:
            print(f"Could not load face cache, rebuilding: {e}")

    root = Path(KNOWN_FACES_DIR)
    if not root.exists():
        print(f"Known faces folder not found: {KNOWN_FACES_DIR}")
        print("Create it like:")
        print("  known_faces/Alex/front.jpg")
        print("  known_faces/Alex/left.jpg")
        return [], []

    names = []
    encodings = []

    image_paths = []
    for suffix in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        image_paths.extend(root.rglob(suffix))

    print(f"Building known face cache from {len(image_paths)} image(s)...")

    for image_path in image_paths:
        try:
            name = face_name_from_path(image_path, root)

            image_rgb = face_recognition.load_image_file(str(image_path))

            face_locations = face_recognition.face_locations(
                image_rgb,
                number_of_times_to_upsample=0,
                model=FACE_DETECTOR_MODEL,
            )

            if not face_locations:
                print(f"No face found in {image_path}")
                continue

            face_encodings = face_recognition.face_encodings(
                image_rgb,
                known_face_locations=face_locations,
                num_jitters=FACE_NUM_JITTERS,
                model="small",
            )

            if not face_encodings:
                print(f"No encoding created for {image_path}")
                continue

            # Use the first face found in each reference photo.
            names.append(name)
            encodings.append(face_encodings[0])
            print(f"Added known face: {name} from {image_path}")

        except Exception as e:
            print(f"Skipping {image_path}: {e}")

    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"names": names, "encodings": encodings}, f)
        print(f"Saved face cache: {cache_path}")
    except Exception as e:
        print(f"Could not save face cache: {e}")

    print(f"Loaded {len(names)} known face encoding(s).")
    return names, encodings


def make_face_key(track_id, class_name, cx, cy, frame_width, frame_height, corridor_position):
    """
    Tracking is enabled by default, so a valid track ID is preferred.
    The zone fallback still keeps face memory separated when no ID is available.
    """
    if track_id != -1:
        return f"face:id:{track_id}"

    x_bin = int(clamp(cx / max(1, frame_width) * 6, 0, 5))
    y_bin = int(clamp(cy / max(1, frame_height) * 4, 0, 3))
    return f"face:{class_name}:{corridor_position}:x{x_bin}:y{y_bin}"


def crop_person_box_with_padding(frame, x1, y1, x2, y2):
    frame_height, frame_width = frame.shape[:2]

    w = box_width(x1, x2)
    h = box_height(y1, y2)
    pad = int(max(w, h) * FACE_CROP_PAD_FRAC)

    left = clamp(int(x1) - pad, 0, frame_width - 1)
    top = clamp(int(y1) - pad, 0, frame_height - 1)
    right = clamp(int(x2) + pad, 0, frame_width - 1)
    bottom = clamp(int(y2) + pad, 0, frame_height - 1)

    if right <= left or bottom <= top:
        return None

    return frame[top:bottom, left:right]



def detect_face_locations_for_emotion(crop_bgr):
    """
    Return face locations as (top, right, bottom, left).

    First tries face_recognition/HOG when installed. If HOG misses the face,
    or face_recognition is not installed, it falls back to OpenCV Haar.
    This lets emotion recognition work independently from identity recognition.
    """
    locations = []

    if crop_bgr is None or crop_bgr.size == 0:
        return locations

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    if face_recognition is not None:
        try:
            locations = face_recognition.face_locations(
                crop_rgb,
                number_of_times_to_upsample=1,
                model=FACE_DETECTOR_MODEL,
            )
        except Exception as e:
            print(f"HOG face detection failed: {e}")
            locations = []

    if locations:
        return locations

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    if HAAR_FACE_CASCADE.empty():
        return []

    faces = HAAR_FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(32, 32),
    )

    for x, y, w, h in faces:
        locations.append((int(y), int(x + w), int(y + h), int(x)))

    return locations


def padded_face_crop(crop_bgr, location, padding_fraction=0.12):
    """Crop one detected face with a little context around the expression."""
    top, right, bottom, left = location
    h_img, w_img = crop_bgr.shape[:2]

    face_w = max(1, right - left)
    face_h = max(1, bottom - top)
    pad = int(max(face_w, face_h) * padding_fraction)

    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(w_img, right + pad)
    bottom = min(h_img, bottom + pad)

    if right <= left or bottom <= top:
        return None

    return crop_bgr[top:bottom, left:right]


def maybe_print_emotion(face_key, face_name, emotion, confidence):
    if not PRINT_EMOTION_TO_TERMINAL:
        return
    if not emotion or confidence is None or emotion == "uncertain":
        return

    now = time.time()
    old = last_emotion_print.get(face_key)

    changed = old is None or old["emotion"] != emotion
    cooldown_passed = old is None or now - old["time"] >= EMOTION_PRINT_COOLDOWN_SECONDS

    if changed or cooldown_passed:
        person_text = face_name or "person"
        print(f"Expression: {person_text} looks {emotion} ({confidence:.0%})")
        last_emotion_print[face_key] = {
            "emotion": emotion,
            "time": now,
        }


def emotion_speech_text(face_name, emotion):
    """Create cautious wording because a facial-expression model can be wrong."""
    spoken_forms = {
        "angry": "angry",
        "disgust": "disgusted",
        "fear": "fearful",
        "happy": "happy",
        "sad": "sad",
        "surprise": "surprised",
        "neutral": "neutral",
    }

    expression = spoken_forms.get(emotion, emotion)
    person_text = face_name or "The person"
    return f"{person_text} looks {expression}."


def update_emotion_speech_candidate(face_key, face_name, emotion, confidence):
    """Return a speech message only after the expression is stable.

    This function is called only on frames where fresh emotion inference ran.
    It does not mark an expression as spoken until commit_emotion_speech() is
    called after the message has actually entered the speech queue.
    """
    if not SPEAK_DETECTED_EMOTIONS:
        return None
    if not emotion or confidence is None or emotion == "uncertain":
        return None
    if emotion == "neutral" and not EMOTION_SPEAK_NEUTRAL:
        return None
    if confidence < EMOTION_SPEAK_MIN_CONFIDENCE:
        return None

    now = time.time()
    mem = emotion_speech_memory.setdefault(face_key, {
        "candidate": None,
        "stable_count": 0,
        "last_spoken_emotion": None,
        "last_spoken_time": 0.0,
    })

    if mem["candidate"] == emotion:
        mem["stable_count"] += 1
    else:
        mem["candidate"] = emotion
        mem["stable_count"] = 1

    if mem["stable_count"] < EMOTION_SPEAK_STABLE_READINGS:
        return None

    same_as_last = mem["last_spoken_emotion"] == emotion
    elapsed = now - mem["last_spoken_time"]

    if same_as_last and elapsed < EMOTION_SPEAK_COOLDOWN_SECONDS:
        return None
    if not same_as_last and mem["last_spoken_time"] > 0:
        if elapsed < EMOTION_SPEAK_CHANGE_GAP_SECONDS:
            return None

    return emotion_speech_text(face_name, emotion)


def commit_emotion_speech(face_key, emotion):
    mem = emotion_speech_memory.get(face_key)
    if mem is None:
        return

    mem["last_spoken_emotion"] = emotion
    mem["last_spoken_time"] = time.time()


def recognize_face_in_person_box(frame_bgr, x1, y1, x2, y2, face_key, frame_count, detection_ran):
    """
    Returns:
      face_name, face_distance_score, emotion, emotion_confidence

    Identity recognition is optional. Emotion recognition continues to work
    even when face_recognition is unavailable or no known-face cache exists.
    """
    identity_available = (
        ENABLE_FACE_RECOGNITION
        and face_recognition is not None
        and len(known_face_encodings) > 0
    )
    emotion_available = (
        ENABLE_EMOTION_RECOGNITION
        and emotion_recognizer is not None
        and emotion_recognizer.enabled
    )

    if not identity_available and not emotion_available:
        return None, None, None, None

    if box_height(y1, y2) < FACE_MIN_PERSON_BOX_HEIGHT:
        return None, None, None, None

    old = face_memory.get(face_key)

    # Identity and emotion have independent schedules. Using one shared minimum
    # interval would accidentally make identity run only when both modulos align.
    identity_due = (
        identity_available
        and detection_ran
        and frame_count % FACE_RECOGNITION_EVERY_N_FRAMES == 0
    )
    emotion_due = (
        emotion_available
        and detection_ran
        and frame_count % EMOTION_EVERY_N_FRAMES == 0
    )

    if not (identity_due or emotion_due):
        if old and time.time() - old["last_seen"] < 3.0:
            return (
                old.get("name"),
                old.get("distance"),
                old.get("emotion"),
                old.get("emotion_confidence"),
            )
        return None, None, None, None

    crop_bgr = crop_person_box_with_padding(frame_bgr, x1, y1, x2, y2)
    if crop_bgr is None:
        return None, None, None, None

    try:
        face_locations = detect_face_locations_for_emotion(crop_bgr)

        if not face_locations:
            # Do not keep refreshing an old identity when the face is no longer
            # visible. This prevents a name or emotion from transferring to a
            # different person who later occupies the same area.
            face_memory.pop(face_key, None)
            emotion_history.pop(face_key, None)
            last_emotion_print.pop(face_key, None)
            return None, None, None, None

        largest_location = max(
            face_locations,
            key=lambda loc: max(1, loc[2] - loc[0]) * max(1, loc[1] - loc[3]),
        )

        face_crop_bgr = padded_face_crop(crop_bgr, largest_location)
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return None, None, None, None

        name = old.get("name") if old else None
        best_distance = old.get("distance") if old else None

        # Identity is performed only when its own schedule is due.
        if identity_due:
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(
                crop_rgb,
                known_face_locations=[largest_location],
                num_jitters=FACE_NUM_JITTERS,
                model="small",
            )

            if encodings:
                distances = face_recognition.face_distance(
                    known_face_encodings,
                    encodings[0],
                )

                if len(distances) > 0:
                    best_index = int(np.argmin(distances))
                    best_distance = float(distances[best_index])
                    name = (
                        known_face_names[best_index]
                        if best_distance <= FACE_MATCH_TOLERANCE
                        else None
                    )

        emotion = old.get("emotion") if old else None
        emotion_confidence = old.get("emotion_confidence") if old else None

        if emotion_due:
            new_emotion, new_confidence = emotion_recognizer.predict(face_crop_bgr)

            if new_emotion is not None and new_confidence is not None:
                emotion, emotion_confidence = smooth_emotion(
                    face_key,
                    new_emotion,
                    new_confidence,
                )

        face_memory[face_key] = {
            "name": name,
            "distance": best_distance,
            "emotion": emotion,
            "emotion_confidence": emotion_confidence,
            "last_seen": time.time(),
        }

        return name, best_distance, emotion, emotion_confidence

    except Exception as e:
        print(f"Face/emotion recognition failed: {e}")
        return None, None, None, None


def cleanup_stale_face_memory():
    now = time.time()

    stale_face_keys = [
        key for key, mem in face_memory.items()
        if now - mem["last_seen"] > TRACK_STALE_SECONDS
    ]

    for key in stale_face_keys:
        del face_memory[key]
        emotion_history.pop(key, None)
        last_emotion_print.pop(key, None)
        emotion_speech_memory.pop(key, None)



# ============================================================
# CAMERA SOURCES
# ============================================================

class OpenCVSource:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def is_opened(self):
        return self.cap.isOpened()

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class LatestFrameSource:
    """Background capture wrapper that exposes only the newest camera frame."""

    def __init__(self, source):
        self.source = source
        self.condition = threading.Condition()
        self.latest_frame = None
        self.sequence = 0
        self.last_returned_sequence = 0
        self.running = True
        self.failed = False
        self.thread = threading.Thread(
            target=self._capture_loop,
            name="latest-camera-frame",
            daemon=True,
        )
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.source.read()
            with self.condition:
                if not ret or frame is None:
                    self.failed = True
                    self.running = False
                    self.condition.notify_all()
                    return
                self.latest_frame = frame
                self.sequence += 1
                self.condition.notify_all()

    def is_opened(self):
        return self.source.is_opened() and not self.failed

    def read(self):
        deadline = time.time() + LATEST_FRAME_WAIT_SECONDS
        with self.condition:
            while (
                self.running
                and self.sequence <= self.last_returned_sequence
                and time.time() < deadline
            ):
                self.condition.wait(timeout=0.10)

            if self.sequence <= self.last_returned_sequence or self.latest_frame is None:
                return False, None

            self.last_returned_sequence = self.sequence
            return True, self.latest_frame.copy()

    def release(self):
        self.running = False
        try:
            self.source.release()
        finally:
            with self.condition:
                self.condition.notify_all()
            self.thread.join(timeout=2.0)


def maybe_wrap_latest_frame(source):
    if USE_LATEST_FRAME_THREAD and source is not None:
        print("Using newest-frame camera capture worker.")
        return LatestFrameSource(source)
    return source


class Picamera2Source:
    def __init__(self):
        from picamera2 import Picamera2
        from libcamera import Transform, controls

        self.picam2 = Picamera2(camera_num=PICAMERA2_CAMERA_INDEX)

        transform = Transform(
            hflip=1 if CAMERA_HFLIP else 0,
            vflip=1 if CAMERA_VFLIP else 0,
        )

        # RGB888 is used directly by this OpenCV pipeline.
        # buffer_count=4 helps avoid frame starvation when YOLO is busy.
        config = self.picam2.create_preview_configuration(
            main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"},
            transform=transform,
            buffer_count=4,
            controls={"FrameRate": CAMERA_FPS},
        )

        self.picam2.configure(config)

        sensor_model = self.picam2.camera_properties.get("Model", "unknown")
        print(f"Picamera2 opened camera index {PICAMERA2_CAMERA_INDEX}: {sensor_model}")

        self.picam2.start()
        time.sleep(0.5)

        # Camera Module 3 uses the IMX708 sensor and supports autofocus.
        # If this is not a Camera Module 3, the control may fail, so keep it safe.
        try:
            if CAMERA3_AF_MODE == "continuous":
                self.picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
                print("Camera Module 3 autofocus: continuous")
            elif CAMERA3_AF_MODE == "auto":
                self.picam2.set_controls({"AfMode": controls.AfModeEnum.Auto})
                print("Camera Module 3 autofocus: auto")
            elif CAMERA3_AF_MODE == "manual":
                self.picam2.set_controls({
                    "AfMode": controls.AfModeEnum.Manual,
                    "LensPosition": float(CAMERA3_MANUAL_LENS_POSITION),
                })
                print(f"Camera Module 3 autofocus: manual LensPosition={CAMERA3_MANUAL_LENS_POSITION}")
            else:
                print("Camera autofocus unchanged/off")
        except Exception as e:
            print(f"Autofocus control was not applied: {e}")

        time.sleep(0.3)

    def is_opened(self):
        return True

    def read(self):
        try:
            frame_bgr = self.picam2.capture_array("main")
            return True, frame_bgr
        except Exception as e:
            print(f"Picamera2 frame read failed: {e}")
            return False, None

    def release(self):
        try:
            self.picam2.stop()
        except Exception:
            pass


def open_source():
    if CAMERA_BACKEND in {"picamera2", "auto"}:
        try:
            print("Trying Picamera2 camera source...")
            src = Picamera2Source()
            print("Using Picamera2.")
            return maybe_wrap_latest_frame(src)
        except Exception as e:
            print(f"Picamera2 not available or failed: {e}")
            if CAMERA_BACKEND == "picamera2":
                return None

    print("Trying OpenCV camera/video source...")
    src = OpenCVSource(OPENCV_SOURCE)
    if src.is_opened():
        print("Using OpenCV source.")
        return maybe_wrap_latest_frame(src)

    return None


# ============================================================
# STANDALONE TOF TEST MODE
# ============================================================

def run_tof_console_test():
    """Print the live 8x8 depth grid without loading camera or YOLO."""
    print("Starting standalone VL53L5CX test.")
    print("Press Ctrl+C to stop. Distances are millimetres; ---- means invalid.")
    start_tof_worker()
    last_counter = -1
    start_time = time.time()

    try:
        while True:
            snapshot = get_tof_snapshot()
            if snapshot["frame_counter"] != last_counter and snapshot["grid_mm"] is not None:
                last_counter = snapshot["frame_counter"]
                grid = snapshot["grid_mm"]
                print("\033[2J\033[H", end="")
                print(
                    "VL53L5CX 8x8 depth grid | "
                    f"age={snapshot['age_seconds']:.3f}s | frame={last_counter}"
                )
                for row in range(8):
                    values = []
                    for col in range(8):
                        value = grid[row, col]
                        values.append(" ----" if not np.isfinite(value) else f"{int(value):5d}")
                    print(" ".join(values))
                print("\nRun the full hat normally after this test succeeds.")

            if snapshot["last_error"] and time.time() - start_time > 2.0:
                print(f"[ToF test] {snapshot['last_error']}")

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nToF test stopped.")
    finally:
        stop_tof_worker()


# ============================================================
# MAIN
# ============================================================

def run_yolo(model, frame):
    if USE_TRACKING:
        return model.track(
            frame,
            persist=True,
            tracker=TRACKER_NAME,
            verbose=False,
            conf=CONFIDENCE,
            imgsz=IMG_SIZE,
            device=DEVICE,
            classes=YOLO_CLASSES_TO_KEEP,
        )

    return model.predict(
        frame,
        verbose=False,
        conf=CONFIDENCE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        classes=YOLO_CLASSES_TO_KEEP,
    )


def main():
    global known_face_names, known_face_encodings, emotion_recognizer

    ensure_log_file()
    start_tof_worker()

    known_face_names, known_face_encodings = load_known_faces()
    emotion_recognizer = EmotionRecognizer(EMOTION_MODEL_FILE)

    print("Loading YOLO model...")
    print(f"Model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    print("Opening source...")
    source = open_source()

    if source is None or not source.is_opened():
        print("Could not open camera/video source.")
        print("For Pi 5 + Camera Module 3, test first with:")
        print("  rpicam-hello --list-cameras")
        print("  rpicam-hello -t 5000 --autofocus-mode continuous")
        print("If those fail, check the 22-pin Pi 5 camera cable orientation and connector latch.")
        stop_tof_worker()
        return

    print()
    print("Whispering Hat Pi 5 Trixie Camera Module 3 version is running.")
    start_speech_worker()
    start_voice_control()
    if SPEAK_STARTUP_MESSAGE:
        play_alert_sound(2, "Whispering hat is ready.")
    print("Walking corridor is ON." if USE_WALKING_CORRIDOR else "Walking corridor is OFF.")
    if ENABLE_TOF:
        if TOF_MOUNT_CALIBRATED:
            print("ToF fusion is ON.")
        else:
            print("ToF is being tested, but fusion is LOCKED until rigid mounting/calibration.")
            print("After mounting, set TOF_MOUNT_CALIBRATED = True.")
    print(
        "Alert history: first warning, escalation, reason changes, and "
        "occasional reminders are spoken."
    )
    print("Press Q to quit if SHOW_WINDOW=True.")
    print("Use Ctrl+C to stop if running headless.")
    print()

    frame_count = 0
    last_results = None

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = source.read()

            if not ret:
                print("No more frames or camera read failed.")
                break

            frame_count += 1

            now = time.time()
            dt = now - prev_time
            if dt > 0:
                fps = 0.90 * fps + 0.10 * (1.0 / dt)
            prev_time = now

            frame_height, frame_width = frame.shape[:2]
            detection_enabled = detection_is_enabled()
            detection_ran = (
                detection_enabled
                and frame_count % PROCESS_EVERY_N_FRAMES == 0
            )

            if not detection_enabled:
                # Keep the live camera/window running, but do zero object/hazard
                # processing until the user says ON again.
                results = None
                last_results = None
            elif detection_ran:
                results = run_yolo(model, frame)
                last_results = results
            else:
                results = last_results

            best_risk = 0
            best_reason = ""
            best_alert_key = None
            best_debug = None
            best_class_name = ""
            best_conf = 0.0
            best_track_id = -1
            best_distance_m = 999.0
            emotion_speech_candidates = []
            nearest_matched_tof_m = float("inf")

            if detection_enabled:
                maybe_print_tof_summary()

            if SHOW_WINDOW:
                draw_walking_corridor(frame)
                if detection_enabled:
                    draw_tof_overlay(frame)

            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        class_name = model.names[cls_id]
                        conf = float(box.conf[0])

                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cx, cy = center_of_box(x1, y1, x2, y2)
                        area = box_area(x1, y1, x2, y2)

                        if box.id is not None:
                            track_id = int(box.id[0])
                        else:
                            track_id = -1

                        foot_y = y2
                        corridor_position = corridor_position_word(cx, foot_y, frame_width, frame_height)

                        alert_key = make_alert_key(
                            track_id=track_id,
                            class_name=class_name,
                            corridor_position=corridor_position,
                        )
                        distance_key = make_distance_key(
                            track_id=track_id,
                            class_name=class_name,
                            corridor_position=corridor_position,
                        )

                        # Only update motion history on real detection frames and if tracking IDs exist.
                        if detection_ran and track_id != -1:
                            track_history[track_id].append({
                                "time": time.time(),
                                "cx": cx,
                                "cy": cy,
                                "area": area,
                                "class_name": class_name,
                            })

                        risk, reason, debug = score_risk(
                            class_name=class_name,
                            cx=cx,
                            cy=cy,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            frame_width=frame_width,
                            frame_height=frame_height,
                            track_id=track_id,
                            distance_key=distance_key,
                            update_distance=detection_ran,
                        )

                        if debug.get("distance_source") == "tof" and debug.get("in_corridor"):
                            nearest_matched_tof_m = min(
                                nearest_matched_tof_m,
                                debug.get("fake_distance_m", float("inf")),
                            )

                        debug["face_name"] = None
                        debug["face_distance_score"] = None
                        debug["emotion"] = None
                        debug["emotion_confidence"] = None

                        if class_name == "person":
                            face_key = make_face_key(
                                track_id=track_id,
                                class_name=class_name,
                                cx=cx,
                                cy=cy,
                                frame_width=frame_width,
                                frame_height=frame_height,
                                corridor_position=corridor_position,
                            )

                            (
                                face_name,
                                face_distance_score,
                                emotion,
                                emotion_confidence,
                            ) = recognize_face_in_person_box(
                                frame_bgr=frame,
                                x1=x1,
                                y1=y1,
                                x2=x2,
                                y2=y2,
                                face_key=face_key,
                                frame_count=frame_count,
                                detection_ran=detection_ran,
                            )

                            if face_name:
                                debug["face_name"] = face_name
                                debug["face_distance_score"] = face_distance_score

                                # Make alerts friendlier:
                                # "Caution. Alex close in path."
                                if reason.startswith("person"):
                                    reason = reason.replace("person", face_name, 1)

                            if emotion:
                                debug["emotion"] = emotion
                                debug["emotion_confidence"] = emotion_confidence
                                maybe_print_emotion(
                                    face_key=face_key,
                                    face_name=face_name,
                                    emotion=emotion,
                                    confidence=emotion_confidence,
                                )

                                # Count only fresh inference frames toward
                                # expression stability, not cached frames.
                                fresh_emotion_frame = (
                                    detection_ran
                                    and frame_count % EMOTION_EVERY_N_FRAMES == 0
                                )
                                close_enough_for_emotion = (
                                    debug["fake_distance_m"]
                                    <= EMOTION_SPEAK_MAX_DISTANCE_M
                                )

                                if fresh_emotion_frame and close_enough_for_emotion:
                                    emotion_message = update_emotion_speech_candidate(
                                        face_key=face_key,
                                        face_name=face_name,
                                        emotion=emotion,
                                        confidence=emotion_confidence,
                                    )
                                    if emotion_message:
                                        emotion_speech_candidates.append({
                                            "face_key": face_key,
                                            "emotion": emotion,
                                            "confidence": emotion_confidence,
                                            "message": emotion_message,
                                        })

                        update_object_memory(alert_key, risk, reason)

                        if LOG_ALL_DETECTIONS and detection_ran:
                            log_row(
                                event="detection",
                                class_name=class_name,
                                confidence=conf,
                                risk=risk,
                                reason=reason,
                                debug=debug,
                                track_id=track_id,
                            )

                        if (
                            risk > best_risk
                            or (risk == best_risk and risk > 0 and debug["fake_distance_m"] < best_distance_m)
                        ):
                            best_risk = risk
                            best_reason = reason
                            best_alert_key = alert_key
                            best_debug = debug
                            best_class_name = class_name
                            best_conf = conf
                            best_track_id = track_id
                            best_distance_m = debug["fake_distance_m"]

                        if SHOW_WINDOW:
                            draw_object_box(
                                frame=frame,
                                x1=x1,
                                y1=y1,
                                x2=x2,
                                y2=y2,
                                class_name=class_name,
                                conf=conf,
                                risk=risk,
                                track_id=track_id,
                                debug=debug,
                            )

            # If real depth reports a close central obstacle but YOLO cannot
            # associate it with a recognized object, warn generically.
            unknown_tof = tof_unknown_path_distance() if detection_enabled else None
            if unknown_tof is not None:
                unknown_distance = unknown_tof["distance_m"]
                unknown_is_separate = (
                    unknown_distance + TOF_UNKNOWN_MATCH_TOLERANCE_M
                    < nearest_matched_tof_m
                )
                if not unknown_is_separate:
                    unknown_tof = None

            if unknown_tof is not None:
                unknown_distance = unknown_tof["distance_m"]
                if unknown_distance <= TOF_UNKNOWN_STOP_M:
                    unknown_risk = 3
                    unknown_reason = "unknown obstacle very close in path"
                elif unknown_distance <= TOF_UNKNOWN_CAUTION_M:
                    unknown_risk = 2
                    unknown_reason = "unknown obstacle close in path"
                elif unknown_distance <= TOF_UNKNOWN_NOTICE_M:
                    unknown_risk = 1
                    unknown_reason = "unknown obstacle in path"
                else:
                    unknown_risk = 0
                    unknown_reason = "unknown obstacle far in path"

                if unknown_risk > 0:
                    unknown_key = "tof:unknown:in_path"
                    unknown_debug = make_unknown_tof_debug(
                        unknown_distance,
                        unknown_tof["valid_cells"],
                    )
                    update_object_memory(unknown_key, unknown_risk, unknown_reason)
                    if (
                        unknown_risk > best_risk
                        or (
                            unknown_risk == best_risk
                            and unknown_distance < best_distance_m
                        )
                    ):
                        best_risk = unknown_risk
                        best_reason = unknown_reason
                        best_alert_key = unknown_key
                        best_debug = unknown_debug
                        best_class_name = "unknown obstacle"
                        best_conf = 1.0
                        best_track_id = -1
                        best_distance_m = unknown_distance

            cleanup_stale_memory()
            cleanup_stale_face_memory()

            safety_message_queued = False

            if best_alert_key is not None:
                print_alert_debug(best_risk, best_reason, best_debug)

                if should_alert(best_alert_key, best_risk, best_reason):
                    message = alert_text(best_risk, best_reason)

                    if message:
                        print(message)
                        safety_message_queued = play_alert_sound(best_risk, message)

                        if best_debug is not None:
                            log_row(
                                event="alert",
                                class_name=best_class_name,
                                confidence=best_conf,
                                risk=best_risk,
                                reason=best_reason,
                                debug=best_debug,
                                track_id=best_track_id,
                            )

            # Announce at most one expression per loop. Skip this frame when a
            # safety warning was just queued, then try again on a later stable
            # inference. Emotion messages use risk 0 and remain low priority.
            if emotion_speech_candidates and not safety_message_queued:
                candidate = max(
                    emotion_speech_candidates,
                    key=lambda item: item["confidence"],
                )
                print(f"Emotion speech: {candidate['message']}")
                if play_alert_sound(0, candidate["message"]):
                    commit_emotion_speech(
                        candidate["face_key"],
                        candidate["emotion"],
                    )

            if SHOW_WINDOW:
                draw_hud(frame, best_risk, best_reason, fps, detection_ran)
                if not detection_enabled:
                    cv2.rectangle(frame, (10, 125), (330, 170), (0, 0, 0), -1)
                    cv2.putText(
                        frame,
                        "DETECTION OFF - say DETECTION ON",
                        (20, 155),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (255, 255, 255),
                        2,
                    )
                cv2.imshow("Whispering Hat Pi Corridor", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), ord("Q")]:
                    break

    except KeyboardInterrupt:
        print()
        print("Stopped by user.")

    finally:
        source.release()
        stop_voice_control()
        stop_tof_worker()
        stop_speech_worker()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--tof-test" in sys.argv:
        run_tof_console_test()
    else:
        main()
