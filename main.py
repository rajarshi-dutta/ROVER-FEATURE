"""
Robo Dog Coordinator
====================
Orchestrates the camera capture pipeline as a single background thread.

camera.py now sends every detected face straight to facerecog.identify_face()
in-memory, before anything is saved. Known faces are discarded on the spot;
only unknown faces get written to disk. Because of that, face recognition is
no longer a separate polling thread — it's initialized once and then called
synchronously inside the camera loop.

Usage:
    python main.py

Install dependencies:
    pip install ultralytics insightface onnxruntime opencv-python numpy imagehash pillow
"""

import threading
import time

# Import the modules
from camera import (
    start_camera_capture, stop_camera_capture,
    get_total_faces_detected, get_total_known_matched,
    get_total_new_unknown, get_total_duplicate_unknown
)
import facerecog
from facerecog import get_known_faces, get_unknown_face_count
from stream_server import start_stream_server


# ============================================================================
# CONFIGURATION
# ============================================================================

CAMERA_THREAD_NAME = "CameraCapture"
STREAM_THREAD_NAME = "StreamServer"

# Status update interval (seconds)
STATUS_UPDATE_INTERVAL = 10

# Web re-streaming server (what the web app connects to)
STREAM_HOST = "0.0.0.0"
STREAM_PORT = 5000

# ============================================================================


class RoboDogCoordinator:
    """Coordinates the camera + inline face-recognition pipeline."""

    def __init__(self):
        self.camera_thread = None
        self.stream_thread = None
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        """Initialize face recognition, then start the camera capture thread."""
        with self.lock:
            if self.running:
                print("[COORDINATOR] Already running!")
                return

            self.running = True

        print("=" * 70)
        print("ROBO DOG COORDINATOR — Starting Background Services")
        print("=" * 70)
        print()

        # Face recognition is loaded up front so the known-faces DB is ready
        # before the first frame ever reaches identify_face().
        if not facerecog.is_ready():
            print("[COORDINATOR] Initializing face recognition…")
            facerecog.init_recognition()

        # Start camera capture thread (does detection + inline identification)
        print("[COORDINATOR] Starting camera capture thread…")
        self.camera_thread = threading.Thread(
            target=start_camera_capture,
            name=CAMERA_THREAD_NAME,
            daemon=True
        )
        self.camera_thread.start()
        time.sleep(1)

        print("[COORDINATOR] Camera thread started.\n")

        # Start the web re-streaming server. It reads frames from the
        # shared buffer camera.py publishes — it never opens its own
        # connection to the ESP32-CAM, so the web app and face detection
        # can run off the single camera connection at the same time.
        print("[COORDINATOR] Starting web stream server…")
        self.stream_thread = threading.Thread(
            target=start_stream_server,
            args=(STREAM_HOST, STREAM_PORT),
            name=STREAM_THREAD_NAME,
            daemon=True
        )
        self.stream_thread.start()
        print(f"[COORDINATOR] Web feed: http://<this-machine-ip>:{STREAM_PORT}/video_feed\n")

        # Start status monitor
        self._monitor_status()

    def _monitor_status(self):
        """Periodically print status of the camera thread."""
        try:
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
                print("STATUS UPDATE")
                print("=" * 70)
                print(f"Camera Thread:        {'🟢 ALIVE' if camera_alive else '🔴 DEAD'}")
                print()
                print(f"Faces Detected:       {faces_detected}")
                print(f"Known Matches:        {known_matched}")
                print(f"New Unknown Saved:    {new_unknown}")
                print(f"Unknown Duplicates:   {duplicates_skipped}")
                print(f"Distinct Unknowns:    {unknown_total}")
                print(f"Known Persons:        {len(known_faces)} {known_faces}")
                print("=" * 70)
                print()

                # Check if the thread died unexpectedly
                if not camera_alive:
                    print("[COORDINATOR] WARNING: The camera thread has died!")
                    self.running = False

        except KeyboardInterrupt:
            pass

    def stop(self):
        """Stop the camera capture thread."""
        with self.lock:
            if not self.running:
                print("[COORDINATOR] Not running!")
                return

            self.running = False

        print("\n[COORDINATOR] Stopping camera thread…")

        stop_camera_capture()

        if self.camera_thread:
            self.camera_thread.join(timeout=5)

        print("[COORDINATOR] Camera thread stopped.")
        print(f"Total faces detected: {get_total_faces_detected()}")
        print(f"Total known matches: {get_total_known_matched()}")
        print(f"Total new unknown faces saved: {get_total_new_unknown()}")
        print(f"Total unknown duplicates skipped: {get_total_duplicate_unknown()}")

    def is_running(self):
        """Check if coordinator is running."""
        return self.running


# ============================================================================
# COMMAND INTERFACE
# ============================================================================

def main():
    """Main entry point with interactive command interface."""
    coordinator = RoboDogCoordinator()

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    ROBO DOG SYSTEM                                  ║
║                 Face Detection & Recognition                        ║
╚══════════════════════════════════════════════════════════════════════╝

Commands:
  start   - Start camera capture + inline face recognition
  stop    - Stop the camera thread
  status  - Show current status
  help    - Show this help
  exit    - Exit program

""")

    try:
        while True:
            cmd = input("[ROBO DOG] Enter command: ").strip().lower()

            if cmd == "start":
                if not coordinator.running:
                    coordinator.start()
                else:
                    print("[ROBO DOG] System is already running!")

            elif cmd == "stop":
                if coordinator.running:
                    coordinator.stop()
                else:
                    print("[ROBO DOG] System is not running!")

            elif cmd == "status":
                if coordinator.running:
                    camera_alive = coordinator.camera_thread and coordinator.camera_thread.is_alive()

                    print()
                    print("=" * 70)
                    print("SYSTEM STATUS")
                    print("=" * 70)
                    print(f"Coordinator:          {'🟢 RUNNING' if coordinator.running else '🔴 STOPPED'}")
                    print(f"Camera Thread:        {'🟢 ALIVE' if camera_alive else '🔴 DEAD'}")
                    print()
                    print(f"Faces Detected:       {get_total_faces_detected()}")
                    print(f"Known Matches:        {get_total_known_matched()}")
                    print(f"New Unknown Saved:    {get_total_new_unknown()}")
                    print(f"Unknown Duplicates:   {get_total_duplicate_unknown()}")
                    print(f"Distinct Unknowns:    {get_unknown_face_count()}")
                    print(f"Known Persons:        {get_known_faces()}")
                    print("=" * 70)
                    print()
                else:
                    print("[ROBO DOG] System is not running. Type 'start' to begin.")

            elif cmd == "help":
                print("""
Commands:
  start   - Start camera capture + inline face recognition
  stop    - Stop the camera thread
  status  - Show current status
  help    - Show this help
  exit    - Exit program
""")

            elif cmd == "exit":
                if coordinator.running:
                    print("[ROBO DOG] Stopping system before exit…")
                    coordinator.stop()
                print("[ROBO DOG] Goodbye!")
                break

            else:
                print(f"[ROBO DOG] Unknown command: '{cmd}'. Type 'help' for available commands.")

    except KeyboardInterrupt:
        print("\n[ROBO DOG] Keyboard interrupt detected.")
        if coordinator.running:
            print("[ROBO DOG] Stopping system…")
            coordinator.stop()
        print("[ROBO DOG] Goodbye!")


if __name__ == "__main__":
    main()