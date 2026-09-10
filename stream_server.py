"""
Stream Server — Re-broadcasts the Camera Feed to the Web App
===============================================================
The ESP32-CAM only allows ONE client to connect to its /stream endpoint
at a time. camera.py is that one client — it reads frames, runs face
detection, AND publishes each frame into a shared in-memory buffer
(camera.get_latest_frame()).

This module reads that buffer (never the camera itself) and re-serves it
as its own MJPEG stream, so any number of web app clients can watch the
feed simultaneously.

Usage:
    from stream_server import start_stream_server
    start_stream_server(host="0.0.0.0", port=5000)

Then point the web app at:
    http://<this-machine-ip>:5000/video_feed

Install:
    pip install flask
"""

import time
import cv2
from flask import Flask, Response

import camera

# ============================================================================
# CONFIGURATION
# ============================================================================

STREAM_FPS = 15                 # target frames/sec sent to web clients
JPEG_QUALITY = 80               # 0-100, higher = better quality/larger size

# ============================================================================

app = Flask(__name__)


def _mjpeg_generator():
    """Yield the shared latest frame as a multipart MJPEG stream."""
    frame_interval = 1.0 / STREAM_FPS

    while True:
        frame = camera.get_latest_frame()

        if frame is not None:
            ok, jpeg = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                )

        time.sleep(frame_interval)


@app.route("/video_feed")
def video_feed():
    """MJPEG endpoint the web app should point its <img>/<video> tag at."""
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/health")
def health():
    """Simple liveness check — reports whether a frame is available yet."""
    frame_available = camera.get_latest_frame() is not None
    return {"status": "ok", "frame_available": frame_available}


def start_stream_server(host="0.0.0.0", port=5000):
    """
    Start the Flask re-streaming server. Blocking — run this in its own
    thread (see main.py), not the main thread, if the camera loop also
    needs to run.
    """
    print(f"[STREAM] Web feed available at http://{host}:{port}/video_feed\n")
    # threaded=True lets Flask serve multiple web clients concurrently
    app.run(host=host, port=port, threaded=True, use_reloader=False)
