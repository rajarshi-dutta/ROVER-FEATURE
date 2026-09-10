"""
Camera Module — Detect & Send Straight to Face Recognition
=============================================================
Runs continuously in a background thread.
Captures frames from ESP32-CAM, detects faces with YOLO, and sends
each face crop DIRECTLY (in memory) to facerecog.identify_face().

camera.py no longer decides what gets saved — facerecog.py does:
    - a known face is discarded immediately
    - a brand-new unknown face is saved to the unknown-faces folder
    - a face that matches one already in that folder is skipped as
      a duplicate (checked by facerecog itself using embeddings,
      not a perceptual hash)

Usage:
    from camera import start_camera_capture

    import threading
    t = threading.Thread(target=start_camera_capture, daemon=True)
    t.start()

Install:
    pip install ultralytics opencv-python insightface onnxruntime
"""

from ultralytics import YOLO
import cv2
import time
import threading
import os
from dotenv import load_dotenv
import facerecog

# ============================================================================
# CONFIGURATION
# ============================================================================
load_dotenv()
MODEL_PATH = "blacknwhite.pt"
SNAPSHOT_URL = os.getenv("videourl")
SNAPSHOT_DELAY = 0.1  # seconds between frames
CONF_THRESHOLD = 0.5

# ============================================================================

# Global state
_model = None
_running = False

# Shared "latest frame" buffer, updated every loop iteration. This is what
# lets a web server re-broadcast the video without opening a second
# connection to the ESP32-CAM (which only supports one client at a time).
_latest_frame = None
_latest_frame_lock = threading.Lock()

# Stats (facerecog.py owns the actual saving/dedup — this just counts outcomes)
_total_faces_detected = 0
_total_faces_lock = threading.Lock()
_total_known_matched = 0
_total_known_lock = threading.Lock()
_total_new_unknown = 0
_total_new_unknown_lock = threading.Lock()
_total_duplicate_unknown = 0
_total_duplicate_lock = threading.Lock()


def _init_model():
    """Initialize YOLO model once."""
    global _model
    if _model is None:
        print("[CAMERA] Loading YOLO model…")
        _model = YOLO(MODEL_PATH)
        print("[CAMERA] Model ready.\n")


def _handle_face(face_crop):
    """
    Send one detected face crop straight to facerecog. facerecog decides
    everything: known vs unknown, and whether an unknown face is new
    (gets saved) or a duplicate of one already in the unknown folder
    (gets skipped). Returns the result dict for logging.
    """
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
    # NO_FACE / ERROR: nothing to count, just falls through to logging

    return result


def _process_frame(frame):
    """
    Detect faces in frame and identify each one directly against
    facerecog. Returns (faces_found, known, new_unknown, duplicate_unknown).
    """
    global _total_faces_detected

    if _model is None:
        return 0, 0, 0, 0

    results = _model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    faces_in_frame = 0
    known_in_frame = 0
    new_unknown_in_frame = 0
    duplicate_in_frame = 0

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        faces_in_frame += 1
        with _total_faces_lock:
            _total_faces_detected += 1

        result = _handle_face(face_crop)

        if result["status"] == "MATCHED":
            known_in_frame += 1
        elif result["status"] == "NEW_UNKNOWN":
            new_unknown_in_frame += 1
        elif result["status"] == "DUPLICATE":
            duplicate_in_frame += 1

    return faces_in_frame, known_in_frame, new_unknown_in_frame, duplicate_in_frame


def start_camera_capture():
    """
    Main camera loop — runs continuously until stopped.
    Connects to ESP32-CAM stream, detects faces, and identifies each
    one inline against facerecog, which handles all saving/dedup.
    """
    global _running, _model, _latest_frame

    _init_model()

    if not facerecog.is_ready():
        facerecog.init_recognition()

    print(f"[CAMERA] Starting capture from {SNAPSHOT_URL}")
    print("[CAMERA] Running in background…\n")

    _running = True
    frame_count = 0

    cap = None
    reconnect_delay = 2

    try:
        while _running:
            try:
                if cap is None or not cap.isOpened():
                    print(f"[CAMERA] Connecting to {SNAPSHOT_URL}…")
                    cap = cv2.VideoCapture(SNAPSHOT_URL)
                    if not cap.isOpened():
                        print(f"[CAMERA] Connection failed, retrying in {reconnect_delay}s…")
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

                # Publish the raw frame for anything else that wants to
                # broadcast it (e.g. stream_server.py) before running
                # detection, so the web feed stays smooth even if detection
                # is momentarily slow.
                with _latest_frame_lock:
                    _latest_frame = frame

                detected, known, new_unknown, duplicate = _process_frame(frame)

                if detected > 0:
                    print(f"[CAMERA] Frame {frame_count} | {frame.shape[1]}x{frame.shape[0]} | "
                          f"Detected: {detected} | Known: {known} | "
                          f"New unknown saved: {new_unknown} | Unknown duplicates skipped: {duplicate}")
                else:
                    print(f"[CAMERA] Frame {frame_count} | {frame.shape[1]}x{frame.shape[0]} | "
                          f"No faces found")

                time.sleep(SNAPSHOT_DELAY)

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
        print(f"\n[CAMERA] Stopped.")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Total faces detected: {_total_faces_detected}")
        print(f"  Total known matches: {_total_known_matched}")
        print(f"  Total new unknown faces saved: {_total_new_unknown}")
        print(f"  Total unknown duplicates skipped: {_total_duplicate_unknown}")


def stop_camera_capture():
    """Stop the camera capture loop."""
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
    """
    Return a copy of the most recent frame read from the camera, or None
    if capture hasn't produced a frame yet. Safe to call from any thread
    (e.g. the web streaming server) — this never opens its own connection
    to the ESP32-CAM.
    """
    with _latest_frame_lock:
        if _latest_frame is None:
            return None
        return _latest_frame.copy()