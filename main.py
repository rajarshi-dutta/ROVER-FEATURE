"""
Robo Dog Coordinator
====================
Orchestrates the camera capture pipeline as a single background thread,
plus a lightweight local Flask server that serves the live feed directly
from the Pi at http://<pi-ip>:5001/stream.

Headless service mode: Controlled via WebSocket commands from the Flask server.
"""

import threading
import time
import socketio

# Import local modules
from camera import (
    start_camera_capture, stop_camera_capture,
    get_total_faces_detected, get_total_known_matched,
    get_total_new_unknown, get_total_duplicate_unknown
)
from camera_server import start_camera_server, stop_camera_server
import facerecog
from facerecog import get_known_faces, get_unknown_face_count


# ============================================================================
# CONFIGURATION
# ============================================================================

CAMERA_THREAD_NAME = "CameraCapture"
STREAM_THREAD_NAME = "CameraServer"
STATUS_UPDATE_INTERVAL = 10

# Windows Laptop Flask Server IP and Port
FLASK_SERVER_URL = "http://10.44.27.21:5050"

# ============================================================================


class RoboDogCoordinator:
    """Coordinates camera capture, face recognition, and the local stream server."""

    def __init__(self):
        self.camera_thread = None
        self.stream_thread = None
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        """Initialize face recognition and start camera and stream threads."""
        with self.lock:
            if self.running:
                print("[COORDINATOR] Already running!")
                return
            self.running = True

        print("=" * 70)
        print("ROBO DOG COORDINATOR — Starting Background Services")
        print("=" * 70)

        # Initialize face recognition models up front
        if not facerecog.is_ready():
            print("[COORDINATOR] Initializing face recognition...")
            facerecog.init_recognition()

        # Start camera capture thread
        print("[COORDINATOR] Starting camera capture thread...")
        self.camera_thread = threading.Thread(
            target=start_camera_capture,
            name=CAMERA_THREAD_NAME,
            daemon=True
        )
        self.camera_thread.start()
        time.sleep(1)
        print("[COORDINATOR] Camera thread started.")

        # Start local MJPEG stream server on port 5001
        print("[COORDINATOR] Starting camera stream server...")
        self.stream_thread = threading.Thread(
            target=start_camera_server,
            name=STREAM_THREAD_NAME,
            daemon=True
        )
        self.stream_thread.start()
        print("[COORDINATOR] Stream available at http://0.0.0.0:5001/stream\n")

        # Start background status logger
        status_thread = threading.Thread(target=self._monitor_status, daemon=True)
        status_thread.start()

    def _monitor_status(self):
        """Periodically log system status to journalctl."""
        while self.running:
            time.sleep(STATUS_UPDATE_INTERVAL)
            if not self.running:
                break

            camera_alive = self.camera_thread and self.camera_thread.is_alive()
            faces_detected = get_total_faces_detected()
            known_matched = get_total_known_matched()
            new_unknown = get_total_new_unknown()
            duplicates_skipped = get_total_duplicate_unknown()
            unknown_total = get_unknown_face_count()
            known_faces = get_known_faces()

            print()
            print("=" * 70)
            print("ROBODOG TELEMETRY UPDATE")
            print("=" * 70)
            print(f"Camera Thread:        {'🟢 ALIVE' if camera_alive else '🔴 DEAD'}")
            print(f"Faces Detected:       {faces_detected}")
            print(f"Known Matches:        {known_matched}")
            print(f"New Unknown Saved:    {new_unknown}")
            print(f"Unknown Duplicates:   {duplicates_skipped}")
            print(f"Distinct Unknowns:    {unknown_total}")
            print(f"Known Persons:        {len(known_faces)} {known_faces}")
            print("=" * 70)
            print()

            if not camera_alive:
                print("[COORDINATOR] WARNING: The camera thread has died!")
                self.running = False

    def stop(self):
        """Stop camera and streaming services."""
        with self.lock:
            if not self.running:
                return
            self.running = False

        print("\n[COORDINATOR] Stopping camera services...")
        stop_camera_capture()
        stop_camera_server()

        if self.camera_thread:
            self.camera_thread.join(timeout=5)

        print("[COORDINATOR] Services stopped cleanly.")


# ============================================================================
# WEBSOCKET CLIENT & REMOTE COMMAND DISPATCHER
# ============================================================================

coordinator = RoboDogCoordinator()
sio = socketio.Client(reconnection=True, reconnection_attempts=0, reconnection_delay=5)


@sio.event
def connect():
    print(f"\n✅ [WEBSOCKET] Connected to Flask Server at {FLASK_SERVER_URL}!")


@sio.event
def disconnect():
    print("\n❌ [WEBSOCKET] Disconnected from Flask Server. Will auto-reconnect...")


@sio.on("robot_command")
def handle_command(data):
    action = data.get("action")
    print(f"📥 [WEBSOCKET] Received command: {action}")

    if action == "forward":
        print("▶ Motors: Moving Forward")
        # motor_forward()
    elif action == "backward":
        print("◀ Motors: Moving Backward")
        # motor_backward()
    elif action == "left":
        print("↺ Motors: Turning Left")
        # motor_left()
    elif action == "right":
        print("↻ Motors: Turning Right")
        # motor_right()
    elif action == "stop":
        print("⏹ Motors: Stopped")
        # motor_stop()
    elif action == "start_patrol":
        print("🚨 Starting Patrol Mode...")
        # start_patrol()


def connect_socket():
    """Background connection loop to ensure Pi reconnects if Flask restarts."""
    while True:
        if not sio.connected:
            try:
                print(f"[WEBSOCKET] Attempting connection to {FLASK_SERVER_URL}...")
                sio.connect(FLASK_SERVER_URL)
            except Exception as e:
                print(f"[WEBSOCKET] Flask server offline ({e}). Retrying in 5s...")
                time.sleep(5)
        else:
            time.sleep(2)


# ============================================================================
# MAIN ENTRYPOINT (HEADLESS / SYSTEMD COMPATIBLE)
# ============================================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    ROBO DOG BACKGROUND SYSTEM                        ║
║                 Face Detection, Stream & Web Control                 ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    # 1. Start camera capture and local HTTP stream (:5001/stream) immediately
    coordinator.start()

    # 2. Start WebSocket client in a resilient background thread
    ws_thread = threading.Thread(target=connect_socket, daemon=True)
    ws_thread.start()

    # 3. Keep main thread alive without blocking on stdin / terminal input
    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        print("\n[ROBO DOG] Shutting down service...")
        coordinator.stop()
        if sio.connected:
            sio.disconnect()
        print("[ROBO DOG] Exit complete.")


if __name__ == "__main__":
    main()