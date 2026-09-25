"""
Camera Module — Raspberry Pi version
====================================
Captures frames from a Raspberry Pi camera, detects faces with YOLO, and
sends each face crop directly (in memory) to facerecog.identify_face().

Backends (set CAMERA_BACKEND):
    "auto"       -> try Pi camera (Picamera2), fall back to USB/OpenCV
    "picamera2"  -> Pi Camera Module (CSI ribbon cable)
    "opencv"     -> USB webcam (index CAMERA_INDEX)

Install (Raspberry Pi OS Bookworm):
    sudo apt install -y python3-picamera2 python3-opencv
    # if using a venv: python3 -m venv --system-site-packages venv
    pip install ultralytics insightface onnxruntime python-dotenv

Usage:
    from camera import start_camera_capture
    import threading
    threading.Thread(target=start_camera_capture, daemon=True).start()
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
FRAME_SIZE = (640, 480)        # (width, height) — keep small for the Pi
CAPTURE_DELAY = 0.03           # seconds between captured frames (smooth video)
DETECT_DELAY = 0.05            # pause between detection passes (lets the Pi breathe)
EXPOSURE_VALUE = 7.0           # Pi cam brightness boost, -8..8 (0 = default)
BRIGHTNESS = 0.1               # -1..1 (0 = default)
CONF_THRESHOLD = 0.4
YOLO_IMGSZ = 320               # smaller = faster on Pi (default is 640)
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
# CAMERA SOURCES (same interface: isOpened / read / release)
# ============================================================================
class _PiCamSource:
    """Pi Camera Module via Picamera2."""

    def __init__(self):
        from picamera2 import Picamera2
        self._cam = Picamera2()
        config = self._cam.create_video_configuration(
            main={"size": FRAME_SIZE, "format": "RGB888"}  # RGB888 = BGR order, OpenCV-ready
        )
        self._cam.configure(config)
        self._cam.set_controls({
            "AeEnable": True,
            "ExposureValue": EXPOSURE_VALUE,
            "Brightness": BRIGHTNESS,
        })
        self._cam.start()
        time.sleep(2)  # let auto-exposure settle
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
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # avoid stale buffered frames
        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # 3 = auto on most UVC cams

    def isOpened(self):
        return self._cap.isOpened()

    def read(self):
        return self._cap.read()

    def release(self):
        self._cap.release()


def _open_camera():
    """Open the camera per CAMERA_BACKEND. Returns a source or None."""
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
# DETECTION
# ============================================================================
def _init_model():
    global _model
    if _model is None:
        print("[CAMERA] Loading YOLO model…")
        _model = YOLO(MODEL_PATH)
        print("[CAMERA] Model ready.\n")


def _handle_face(face_crop):
    global _total_known_matched, _total_new_unknown, _total_duplicate_unknown

    result = facerecog.identify_face(face_crop)

    if result["status"] == "MATCHED":
        with _total_known_lock:
            _total_known_matched += 1
    elif result["status"] == "NEW_UNKNOWN":
        with _total_new_unknown_lock:
            _total_new_unknown += 1
    elif result["status"] == "DUPLICATE":
        with _total_duplicate_lock:
            _total_duplicate_unknown += 1

    return result


def _process_frame(frame):
    """Returns (faces_found, known, new_unknown, duplicate_unknown)."""
    global _total_faces_detected

    if _model is None:
        return 0, 0, 0, 0

    results = _model.predict(source=frame, conf=CONF_THRESHOLD,
                             imgsz=YOLO_IMGSZ, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    faces = known = new_unknown = duplicate = 0
    h, w = frame.shape[:2]

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        faces += 1
        with _total_faces_lock:
            _total_faces_detected += 1

        result = _handle_face(face_crop)

        if result["status"] == "MATCHED":
            known += 1
        elif result["status"] == "NEW_UNKNOWN":
            new_unknown += 1
        elif result["status"] == "DUPLICATE":
            duplicate += 1

    return faces, known, new_unknown, duplicate


def _detect_loop():
    """Runs YOLO + face recognition on the newest frame, in its own thread,
    so slow detection never slows the video."""
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
    """Capture loop (fast, smooth video) + a separate detection thread."""
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
    """Copy of the latest frame, or None. Thread-safe."""
    with _latest_frame_lock:
        if _latest_frame is None:
            return None
        return _latest_frame.copy()
