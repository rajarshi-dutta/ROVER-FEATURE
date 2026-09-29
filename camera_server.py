"""
Camera Server — Serves the Live Feed Directly from the Pi
=============================================================
Runs a small Flask server on the Pi itself, reading frames from
camera.py's shared buffer (camera.get_latest_frame()) — it never
opens its own connection to the camera.

Exposes:
    GET /stream   -> MJPEG stream (multipart/x-mixed-replace)
                     Point an <img src="..."> tag straight at this,
                     from any webpage on your network:
                         <img src="http://<pi-ip>:5001/stream">
    GET /status   -> {"online": true/false}

No token, no relay through another backend — this is just a normal
Flask MJPEG server, same pattern used by most Pi camera projects.

Usage:
    from camera_server import start_camera_server, stop_camera_server

    import threading
    t = threading.Thread(target=start_camera_server, daemon=True)
    t.start()

Install:
    pip install flask
"""

import time
import cv2
from flask import Flask, Response, jsonify

import camera

# ============================================================================
# CONFIGURATION
# ============================================================================
HOST = "0.0.0.0"   # listen on all interfaces so other devices on the LAN can reach it
PORT = 5001
JPEG_QUALITY = 70
STREAM_FPS_DELAY = 0.1  # seconds between frames sent to a connected viewer
STALE_FRAME_TIMEOUT = 5  # seconds since last real frame before /status reports offline

# ============================================================================

app = Flask(__name__)
_encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
_last_frame_seen_at = 0


def _mjpeg_generator():
    """Yields JPEG frames as a multipart HTTP stream, pulling the latest
    frame from camera.py's shared buffer each time."""
    global _last_frame_seen_at

    while True:
        frame = camera.get_latest_frame()

        if frame is not None:
            _last_frame_seen_at = time.time()
            ok, jpeg = cv2.imencode(".jpg", frame, _encode_params)
            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                )

        time.sleep(STREAM_FPS_DELAY)


@app.route("/stream")
def stream():
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def status():
    online = (
        _last_frame_seen_at > 0
        and (time.time() - _last_frame_seen_at) < STALE_FRAME_TIMEOUT
    )
    return jsonify({"online": online})


def start_camera_server():
    """Run the Flask server. Blocks until the process exits — call this
    in its own thread."""
    print(f"[CAMERA SERVER] Serving stream at http://<pi-ip>:{PORT}/stream")
    # threaded=True so /status and /stream can be handled concurrently by
    # multiple viewers without blocking each other.
    app.run(host=HOST, port=PORT, threaded=True, debug=False, use_reloader=False)


def stop_camera_server():
    """
    Flask's built-in dev server has no clean programmatic stop when run
    this way. Since this thread is started as daemon=True in main.py, it
    is killed automatically when the main program exits — nothing to do
    here, this function exists only so main.py's shutdown sequence has a
    consistent stop_* call to make (matching camera.py's pattern).
    """
    pass