"""
Mock ESP32-CAM Server
----------------------
Simulates an ESP32-CAM module by broadcasting an MJPEG video stream over
your local network, at the same kind of URL the real hardware uses
(http://<ip>:81/stream). Point your existing OpenCV client code at this
server's address and it behaves exactly like a real ESP32-CAM feed.

Usage:
    1. Set SOURCE below to a video file path, or 0 for your desktop webcam.
    2. Run: python mock_esp32_cam.py
    3. It prints the URL to use, e.g. http://192.168.1.23:81/stream
    4. In your existing YOLO/OpenCV script, set:
         stream_url = "http://<printed-ip>:81/stream"
       ...and everything else in your pipeline works unchanged.

Requires: pip install flask opencv-python
"""

import cv2
import socket
import time
from flask import Flask, Response

# ── CONFIG ────────────────────────────────────────────────
SOURCE = "C:/Users/Rajarshi/OneDrive/Videos/realme videos/VID20260412102025.mp4"   # path to a video file, or 0 for webcam
LOOP_VIDEO = True           # restart video from the beginning when it ends
FPS_LIMIT = 30               # cap streaming rate (set None for no limit)
PORT = 81                    # matches ESP32-CAM's default stream port
# ──────────────────────────────────────────────────────────

app = Flask(__name__)


def get_local_ip():
    """Best-effort way to find the machine's LAN IP for display purposes."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def generate_frames():
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {SOURCE}")

    frame_delay = (1.0 / FPS_LIMIT) if FPS_LIMIT else 0

    while True:
        ret, frame = cap.read()

        if not ret:
            if LOOP_VIDEO and SOURCE != 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        frame = cv2.resize(frame, (640, 480))  # ← add this
        ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])  # ← lower quality
        if not ok:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        if frame_delay:
            time.sleep(frame_delay)

    cap.release()


@app.route('/stream')
def stream():
    return Response(generate_frames(),
                     mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return """
    <html>
      <body style="font-family: sans-serif; text-align:center;">
        <h2>Mock ESP32-CAM</h2>
        <img src="/stream" width="720">
      </body>
    </html>
    """


if __name__ == '__main__':
    ip = get_local_ip()
    print("=" * 50)
    print("Mock ESP32-CAM server starting...")
    print(f"Broadcasting source: {SOURCE}")
    print(f"View in browser:      http://{ip}:{PORT}/")
    print(f"Use in OpenCV/YOLO:   http://{ip}:{PORT}/stream")
    print("=" * 50)
    app.run(host='0.0.0.0', port=PORT, threaded=True)
