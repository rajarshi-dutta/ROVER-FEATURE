"""
Camera Pusher — Sends the Camera Feed to the Render Backend
===============================================================
Replaces the old stream_server.py. Instead of running its own local
Flask server (only reachable on the home network), this module reads
the shared frame buffer from camera.py and POSTs each frame up to the
Render backend, which re-serves it to the web dashboard from anywhere.

Usage:
    from camera_pusher import start_camera_pusher
    start_camera_pusher()

Install:
    pip install requests python-dotenv
"""

import time
import os
import cv2
import requests
from dotenv import load_dotenv

import camera

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Your Render backend's base URL (no trailing slash)
BACKEND_URL = os.getenv("BACKEND_URL", "https://robodog-web.onrender.com/api/robot")

# Must match CAMERA_PUSH_TOKEN set in Render's environment variables
CAMERA_PUSH_TOKEN = os.getenv("CAMERA_PUSH_TOKEN", "change-me")

PUSH_FPS = 10                    # frames/sec sent to the backend (keep modest — bandwidth)
JPEG_QUALITY = 70                # 0-100, lower = smaller/faster upload
REQUEST_TIMEOUT = 5              # seconds — don't let a slow upload pile up

# ============================================================================


def start_camera_pusher():
    """
    Blocking loop — run this in its own thread (see main.py), same as the
    old start_stream_server. Reads camera.get_latest_frame() and POSTs it.
    """
    frame_interval = 1.0 / PUSH_FPS
    url = f"{BACKEND_URL}/camera/frame"
    headers = {
        "Content-Type": "image/jpeg",
        "X-Camera-Token": CAMERA_PUSH_TOKEN,
    }

    print(f"[PUSHER] Pushing camera feed to {url}\n")

    consecutive_failures = 0

    while True:
        start = time.time()
        frame = camera.get_latest_frame()

        if frame is not None:
            ok, jpeg = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )

            if ok:
                try:
                    resp = requests.post(
                        url,
                        data=jpeg.tobytes(),
                        headers=headers,
                        timeout=REQUEST_TIMEOUT,
                    )

                    if resp.status_code == 200:
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        print(f"[PUSHER] Backend rejected frame: {resp.status_code} {resp.text}")

                except requests.exceptions.RequestException as e:
                    consecutive_failures += 1
                    if consecutive_failures % 20 == 1:  # don't spam logs every single failure
                        print(f"[PUSHER] Upload failed: {e}")

        elapsed = time.time() - start
        sleep_time = max(0, frame_interval - elapsed)
        time.sleep(sleep_time)
