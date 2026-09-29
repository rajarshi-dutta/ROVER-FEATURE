"""
Camera Module — Raspberry Pi version (Full-Frame Processing)
============================================================
Captures frames from a Raspberry Pi camera, runs YOLO to track/log telemetry,
and passes the full uncropped frame directly to facerecog.identify_face().
"""

from ultralytics import YOLO
import cv2
import time
import threading
from dotenv import load_dotenv
import facerecog

# ============================================================================
# CONFIGURATION
# ============================================================================
load_dotenv()
MODEL_PATH = "blacknwhite.pt"
CAMERA_BACKEND = "auto"        # "auto" | "picamera2" | "opencv"
CAMERA_INDEX = 0               # only used for the OpenCV/USB backend
FRAME_SIZE = (1280, 720)       # (width, height) — 720p HD resolution
CAPTURE_DELAY = 0.00           # seconds between captured frames
DETECT_DELAY = 0.00            # pause between detection passes
EXPOSURE_VALUE = 7.0           # Pi cam brightness boost
BRIGHTNESS = 0.1               # -1..1
CONF_THRESHOLD = 0.4
YOLO_IMGSZ = 416
# ============================================================================

_model = None
_running = False

_latest_frame = None
_latest_frame_id = 0
_latest_frame_lock = threading.Lock()

_total_faces_detected = 0
_total_faces_lock = threading.Lock()
_total_known_matched = 0
_total_known_lock = threading.Lock()
_total_new_unknown = 0
_total_new_unknown_lock = threading.Lock()
_total_duplicate_unknown = 0
_total_duplicate_lock = threading.Lock()


# ============================================================================
# CAMERA SOURCES
# ============================================================================
class _PiCamSource:
    """Pi Camera Module via Picamera2."""

    def __init__(self):
        from picamera2 import Picamera2
        self._cam = Picamera2()
        config = self._cam.create_video_configuration(
            main={"size": FRAME_SIZE, "format": "RGB888"}
        )
        self._cam.configure(config)
        self._cam.set_controls({
            "AeEnable": True,
            "ExposureValue": EXPOSURE_VALUE,
            "Brightness": BRIGHTNESS,
        })
        self._cam.start()
        time.sleep(2)
        self._open = True

    def isOpened(self):
        return self._open

    def read(self):
        try:
            return True, self._cam.capture_array()
        except Exception:
            return False, None

    def release(self):
        if self._open:
            self._open = False
            try:
                self._cam.stop()
                self._cam.close()
            except Exception:
                pass


class _CvSource:
    """USB webcam via OpenCV."""

    def __init__(self):
        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

    def isOpened(self):
        return self._cap.isOpened()

    def read(self):
        return self._cap.read()

    def release(self):
        self._cap.release()


def _open_camera():
    if CAMERA_BACKEND in ("auto", "picamera2"):
        try:
            src = _PiCamSource()
            print("[CAMERA] Using Pi camera (Picamera2).")
            return src
        except Exception as e:
            print(f"[CAMERA] Picamera2 unavailable: {e}")
            if CAMERA_BACKEND == "picamera2":
                return None

    src = _CvSource()
    if src.isOpened():
        print(f"[CAMERA] Using OpenCV camera index {CAMERA_INDEX}.")
        return src
    src.release()
    return None


# ============================================================================
# DETECTION & FULL-FRAME IDENTIFICATION
# ============================================================================
def _init_model():
    global _model
    if _model is None:
        print("[CAMERA] Loading YOLO model…")
        _model = YOLO(MODEL_PATH)
        print("[CAMERA] Model ready.\n")


def _process_frame(frame):
    """Runs YOLO for tracking boxes, but passes the FULL frame to facerecog."""
    global _total_faces_detected, _total_known_matched, _total_new_unknown, _total_duplicate_unknown

    if _model is None:
        return 0, 0, 0, 0

    # Optional: YOLO prediction check to confirm a face/person is present in frame
    results = _model.predict(source=frame, conf=CONF_THRESHOLD, imgsz=YOLO_IMGSZ, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    faces = known = new_unknown = duplicate = 0

    if len(boxes) > 0:
        faces = len(boxes)
        with _total_faces_lock:
            _total_faces_detected += faces

        # Pass the FULL frame directly to InsightFace instead of tight crops
        result = facerecog.identify_face(frame)

        status = result.get("status")
        if status == "MATCHED":
            known += 1
            with _total_known_lock:
                _total_known_matched += 1
        elif status == "NEW_UNKNOWN":
            new_unknown += 1
            with _total_new_unknown_lock:
                _total_new_unknown += 1
        elif status == "DUPLICATE":
            duplicate += 1
            with _total_duplicate_lock:
                _total_duplicate_unknown += 1

    return faces, known, new_unknown, duplicate


def _detect_loop():
    last_id = -1
    passes = 0
    while _running:
        with _latest_frame_lock:
            frame = None if _latest_frame is None else _latest_frame.copy()
            fid = _latest_frame_id

        if frame is None or fid == last_id:
            time.sleep(0.02)
            continue
        last_id = fid

        try:
            detected, known, new_unknown, duplicate = _process_frame(frame)
            passes += 1
            if detected > 0:
                print(f"[CAMERA] Pass {passes} | Detected: {detected} | Known: {known} | "
                      f"New unknown saved: {new_unknown} | Duplicates skipped: {duplicate}")
        except Exception as e:
            print(f"[CAMERA] Detection error: {e}")

        time.sleep(DETECT_DELAY)


def start_camera_capture():
    global _running, _latest_frame, _latest_frame_id

    _init_model()

    if not facerecog.is_ready():
        facerecog.init_recognition()

    print("[CAMERA] Starting capture…")
    print("[CAMERA] Running in background…\n")

    _running = True
    frame_count = 0
    cap = None
    reconnect_delay = 2

    threading.Thread(target=_detect_loop, name="Detect", daemon=True).start()

    try:
        while _running:
            try:
                if cap is None or not cap.isOpened():
                    cap = _open_camera()
                    if cap is None:
                        print(f"[CAMERA] No camera found, retrying in {reconnect_delay}s…")
                        time.sleep(reconnect_delay)
                        continue

                ret, frame = cap.read()

                if not ret or frame is None:
                    print("[CAMERA] Failed to read frame, reconnecting…")
                    cap.release()
                    cap = None
                    time.sleep(1)
                    continue

                frame_count += 1
                with _latest_frame_lock:
                    _latest_frame = frame
                    _latest_frame_id = frame_count

                time.sleep(CAPTURE_DELAY)

            except Exception as e:
                print(f"[CAMERA] Error: {e}")
                if cap:
                    cap.release()
                cap = None
                time.sleep(reconnect_delay)

    except KeyboardInterrupt:
        print("\n[CAMERA] Interrupted by user.")

    finally:
        _running = False
        if cap:
            cap.release()
        print("\n[CAMERA] Stopped.")
        print(f"  Total frames captured: {frame_count}")
        print(f"  Total faces detected: {_total_faces_detected}")
        print(f"  Total known matches: {_total_known_matched}")
        print(f"  Total new unknown faces saved: {_total_new_unknown}")
        print(f"  Total unknown duplicates skipped: {_total_duplicate_unknown}")


def stop_camera_capture():
    global _running
    _running = False


def get_total_faces_detected():
    with _total_faces_lock:
        return _total_faces_detected


def get_total_known_matched():
    with _total_known_lock:
        return _total_known_matched


def get_total_new_unknown():
    with _total_new_unknown_lock:
        return _total_new_unknown


def get_total_duplicate_unknown():
    with _total_duplicate_lock:
        return _total_duplicate_unknown


def get_model_status():
    return _model is not None


def get_latest_frame():
    with _latest_frame_lock:
        if _latest_frame is None:
            return None
        return _latest_frame.copy()