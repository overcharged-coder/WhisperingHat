#!/usr/bin/env python3
import time
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import controls

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter


BASE = Path(__file__).resolve().parent
MODEL = BASE / "emotion_model.tflite"
CASCADE = Path("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

if not MODEL.exists():
    raise SystemExit(f"ERROR: model not found: {MODEL}")

if not CASCADE.exists():
    raise SystemExit(
        f"ERROR: Haar cascade not found: {CASCADE}\n"
        "Run: sudo apt update && sudo apt install -y opencv-data"
    )

detector = cv2.CascadeClassifier(str(CASCADE))
if detector.empty():
    raise SystemExit(f"ERROR: could not load Haar cascade: {CASCADE}")

interpreter = Interpreter(model_path=str(MODEL), num_threads=2)
interpreter.allocate_tensors()

input_info = interpreter.get_input_details()[0]
output_info = interpreter.get_output_details()[0]

shape = input_info["shape"]
input_h = int(shape[1])
input_w = int(shape[2])
input_c = int(shape[3])

print(f"MODEL OK: {MODEL}", flush=True)
print(f"INPUT: {tuple(shape)} {input_info['dtype']}", flush=True)
print(f"OUTPUT: {tuple(output_info['shape'])} {output_info['dtype']}", flush=True)
print(f"CASCADE OK: {CASCADE}", flush=True)
print("Starting camera. Look directly at it. Ctrl+C to stop.", flush=True)

camera = Picamera2()
config = camera.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    buffer_count=4,
    controls={"FrameRate": 15},
)
camera.configure(config)
camera.start()

try:
    camera.set_controls({"AfMode": controls.AfModeEnum.Continuous})
except Exception:
    pass

last_message = 0.0

try:
    while True:
        rgb = camera.capture_array("main")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(40, 40),
        )

        now = time.time()

        if len(faces) == 0:
            if now - last_message >= 1.0:
                print("NO FACE DETECTED", flush=True)
                last_message = now
            continue

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        pad = int(max(w, h) * 0.12)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(gray.shape[1], x + w + pad)
        y2 = min(gray.shape[0], y + h + pad)

        face = gray[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face = cv2.resize(face, (input_w, input_h), interpolation=cv2.INTER_AREA)

        if input_c == 1:
            face = face[..., np.newaxis]
        else:
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)

        dtype = input_info["dtype"]

        if np.issubdtype(dtype, np.floating):
            tensor = face.astype(np.float32)
            tensor = (tensor / 255.0 - 0.5) * 2.0
        else:
            scale, zero_point = input_info.get("quantization", (0.0, 0))
            if scale and scale > 0:
                normalized = (face.astype(np.float32) / 255.0 - 0.5) * 2.0
                tensor = np.round(normalized / scale + zero_point)
                info = np.iinfo(dtype)
                tensor = np.clip(tensor, info.min, info.max)
            else:
                tensor = face

        tensor = np.expand_dims(tensor, axis=0).astype(dtype)

        interpreter.set_tensor(input_info["index"], tensor)
        interpreter.invoke()

        scores = interpreter.get_tensor(output_info["index"])[0]
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        out_scale, out_zero = output_info.get("quantization", (0.0, 0))
        if not np.issubdtype(output_info["dtype"], np.floating) and out_scale:
            scores = (scores - out_zero) * out_scale

        if (
            np.any(scores < 0)
            or np.any(scores > 1)
            or not np.isclose(float(scores.sum()), 1.0, atol=0.15)
        ):
            scores = np.exp(scores - np.max(scores))
            scores = scores / max(float(scores.sum()), 1e-8)
        else:
            scores = scores / max(float(scores.sum()), 1e-8)

        best = int(np.argmax(scores))
        label = LABELS[best] if best < len(LABELS) else f"class_{best}"

        if now - last_message >= 0.75:
            print(
                f"EMOTION: {label} {scores[best] * 100:.1f}% "
                f"| face={w}x{h}px",
                flush=True,
            )
            last_message = now

except KeyboardInterrupt:
    print("\nStopped.", flush=True)
finally:
    camera.stop()
