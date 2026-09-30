import os
import time
import signal
import sys
import threading
import serial

from gpiozero import DistanceSensor, Servo

try:
    import socketio            # pip install "python-socketio[client]" websocket-client
except ImportError:
    socketio = None


# ============================================================
# SAFETY CONFIGURATION
# ============================================================

SAFE_DISTANCE_CM = 30.0
MAX_SENSOR_DISTANCE_M = 2.0
SERVO_SETTLE_TIME = 0.4
LOOP_DELAY = 0.1

# Arduino stops the motors by itself if it hears nothing for this long.
ARDUINO_WATCHDOG_SECONDS = 2.0

# Resend the current movement command this often to keep that watchdog alive.
COMMAND_REFRESH_INTERVAL = 0.25

# MANUAL MODE safety: refuse to drive FORWARD closer than this to an obstacle.
ENABLE_MANUAL_OBSTACLE_STOP = True
MANUAL_SAFETY_STOP_CM = 15.0


# ============================================================
# RASPBERRY PI GPIO CONFIGURATION
# ============================================================

ULTRASONIC_TRIGGER_PIN = 23
ULTRASONIC_ECHO_PIN = 24
SERVO_PIN = 18

ARDUINO_SERIAL_PORT = "/dev/serial0"   # USB cable instead? use "/dev/ttyUSB0" or "/dev/ttyACM0"
ARDUINO_BAUD_RATE = 115200   # must match Serial.begin() in the Arduino sketch


# ============================================================
# WEBSITE / BACKEND LINK
# ============================================================

# Address of the Flask backend the website talks to.
# The Pi must be able to reach it, so use the backend PC's LAN IP,
# e.g.  export BACKEND_URL=http://192.168.1.50:5050
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:5050")

# Optional: token the backend expects from the robot (sent as a Bearer header).
ROBOT_TOKEN = os.environ.get("ROBOT_TOKEN", "")

# Socket.IO event names (must match app.py).
CONTROL_EVENT = "robot_command"   # app.py emits this on every /api/control call
MODE_EVENT = "mode"


# ============================================================
# SERVO POSITIONS
# ============================================================

LEFT_POSITION = -0.8
CENTER_POSITION = 0.0
RIGHT_POSITION = 0.8


# ============================================================
# MOVEMENT TIMINGS
# ============================================================

BACKWARD_TIME = 0.7
TURN_LEFT_TIME = 0.6
TURN_RIGHT_TIME = 0.6
FORWARD_AFTER_SCAN_TIME = 0.5


# ============================================================
# GLOBAL STATE
# ============================================================

sensor = None
servo = None
arduino = None
sio = None

running = True

current_command = "S"
last_command_time = 0.0

# MODE TOGGLE
#   True  -> autonomous navigation, website controller IGNORED
#   False -> website controller drives the rover, autonomy paused
# Starts True so the rover behaves as before until the website says otherwise.
autonomous_mode = True

serial_lock = threading.Lock()
mode_lock = threading.Lock()

ACTION_TO_COMMAND = {
    "forward": "F",
    "backward": "B",
    "left": "L",
    "right": "R",
    "stop": "S",
}


# ============================================================
# SERIAL COMMUNICATION
# ============================================================

def connect_to_arduino():
    global arduino

    try:
        print(f"[SERIAL] Connecting to Arduino on {ARDUINO_SERIAL_PORT}...")

        arduino = serial.Serial(
            port=ARDUINO_SERIAL_PORT,
            baudrate=ARDUINO_BAUD_RATE,
            timeout=1,
            write_timeout=1      # never let a stuck port freeze the rover loop
        )

        time.sleep(2.0)   # a Nano resets when a USB serial port opens
        print("[SERIAL] Arduino connected.")
        return True

    except Exception as e:
        print(f"[SERIAL] Could not connect to Arduino: {e}")
        arduino = None
        return False


def send_command(command):
    """
    Send a movement command to the Arduino.
    F = Forward, B = Backward, L = Left, R = Right, S = Stop
    Safe to call from any thread.
    """

    global current_command
    global last_command_time

    if command not in ("F", "B", "L", "R", "S"):
        print(f"[SERIAL] Invalid command: {command}")
        return False

    current_command = command
    last_command_time = time.monotonic()

    if arduino is None:
        print(f"[SERIAL] Arduino unavailable. Command {command} not sent.")
        return False

    try:
        with serial_lock:
            arduino.write((command + "\n").encode("ascii"))
            arduino.flush()

        print(f"[SERIAL] → Arduino: {command}")
        return True

    except Exception as e:
        print(f"[SERIAL] Send error: {e}")
        return False


def refresh_command():
    """Resend the current command so the Arduino watchdog stays quiet."""

    if current_command == "S":
        return

    if time.monotonic() - last_command_time >= COMMAND_REFRESH_INTERVAL:
        send_command(current_command)


# ============================================================
# MOTOR COMMAND FUNCTIONS (autonomous)
# ============================================================

def motor_forward():
    print("[AUTO] Moving Forward...")
    send_command("F")


def motor_backward():
    print("[AUTO] Reversing...")
    send_command("B")


def motor_turn_left():
    print("[AUTO] Turning Left...")
    send_command("L")


def motor_turn_right():
    print("[AUTO] Turning Right...")
    send_command("R")


def motor_stop():
    print("[AUTO] Stopping.")
    send_command("S")


# ============================================================
# HARDWARE INITIALIZATION
# ============================================================

def initialize_hardware():
    global sensor
    global servo

    print("[INIT] Initializing HC-SR04...")

    sensor = DistanceSensor(
        echo=ULTRASONIC_ECHO_PIN,
        trigger=ULTRASONIC_TRIGGER_PIN,
        max_distance=MAX_SENSOR_DISTANCE_M
    )

    print("[INIT] HC-SR04 initialized.")
    print("[INIT] Initializing servo...")

    servo = Servo(SERVO_PIN)
    servo.value = CENTER_POSITION

    print("[INIT] Servo initialized.")

    connect_to_arduino()

    # Always begin with motors stopped.
    send_command("S")
    time.sleep(0.5)


# ============================================================
# ULTRASONIC SENSOR
# ============================================================

def get_distance_cm():
    """
    Read the HC-SR04 distance in cm.
    A sensor failure is treated as an unsafe (blocked) condition.
    """

    if sensor is None:
        return 0.0

    try:
        distance_cm = sensor.distance * 100.0
        return 0.0 if distance_cm < 0 else distance_cm

    except Exception as e:
        print(f"[AUTO] Ultrasonic sensor error: {e}")
        return 0.0


# ============================================================
# SERVO SCANNING
# ============================================================

def scan_direction(position, name):
    if servo is None:
        return 0.0

    print(f"[SCAN] Looking {name}...")

    try:
        servo.value = position
        time.sleep(SERVO_SETTLE_TIME)

        distance = get_distance_cm()
        print(f"[SCAN] {name}: {distance:.1f} cm")
        return distance

    except Exception as e:
        print(f"[SCAN] Servo/sensor error while looking {name}: {e}")
        return 0.0


def scan_environment():
    print("\n[SCAN] Scanning environment...")

    left_distance = scan_direction(LEFT_POSITION, "LEFT")
    center_distance = scan_direction(CENTER_POSITION, "CENTER")
    right_distance = scan_direction(RIGHT_POSITION, "RIGHT")

    if servo is not None:
        servo.value = CENTER_POSITION

    time.sleep(0.2)

    print(
        f"[SCAN] Result -> "
        f"L: {left_distance:.1f} cm | "
        f"C: {center_distance:.1f} cm | "
        f"R: {right_distance:.1f} cm"
    )

    return left_distance, center_distance, right_distance


# ============================================================
# DIRECTION DECISION
# ============================================================

def choose_direction(left, center, right):
    print("[AUTO] Choosing navigation direction...")

    if center > SAFE_DISTANCE_CM:
        print("[AUTO] Center is clear -> FORWARD")
        return "forward"

    print("[AUTO] Center blocked!")

    if left > SAFE_DISTANCE_CM and left > right:
        print(f"[AUTO] Left is clearer ({left:.1f} cm) -> LEFT")
        return "left"

    if right > SAFE_DISTANCE_CM and right > left:
        print(f"[AUTO] Right is clearer ({right:.1f} cm) -> RIGHT")
        return "right"

    if left > right:
        print("[AUTO] Both sides restricted. Left has more space -> LEFT")
        return "left"

    if right > left:
        print("[AUTO] Both sides restricted. Right has more space -> RIGHT")
        return "right"

    print("[AUTO] No safe direction found -> BACKWARD")
    return "backward"


# ============================================================
# OBSTACLE AVOIDANCE
# ============================================================

def run_timed_move(start_fn, duration):
    """
    Start a movement, keep the Arduino watchdog fed for `duration`
    seconds, then stop.  Aborts immediately if the mode is switched
    to manual (the mode switch has already stopped the motors).
    """

    start_fn()

    end_time = time.monotonic() + duration

    while time.monotonic() < end_time and running and autonomous_mode:
        refresh_command()
        time.sleep(0.05)

    if autonomous_mode:
        motor_stop()


def avoid_obstacle():
    print("\n⚠️ [AUTO] Obstacle detected!")

    motor_stop()
    time.sleep(0.2)

    left, center, right = scan_environment()

    # Mode may have been switched to manual while scanning.
    if not autonomous_mode:
        print("[AUTO] Mode changed to manual during scan. Aborting.")
        return

    direction = choose_direction(left, center, right)

    if direction == "forward":
        run_timed_move(motor_forward, FORWARD_AFTER_SCAN_TIME)
    elif direction == "left":
        run_timed_move(motor_turn_left, TURN_LEFT_TIME)
    elif direction == "right":
        run_timed_move(motor_turn_right, TURN_RIGHT_TIME)
    elif direction == "backward":
        run_timed_move(motor_backward, BACKWARD_TIME)


# ============================================================
# MODE TOGGLE + WEBSITE CONTROLLER
# ============================================================

def set_autonomous(value):
    """
    Switch between autonomous (True) and controller (False) mode.
    Motors are always stopped when the mode actually changes.
    """

    global autonomous_mode

    value = bool(value)

    with mode_lock:
        changed = value != autonomous_mode
        autonomous_mode = value

    if changed:
        send_command("S")

        if value:
            print("[MODE] AUTONOMOUS ON  -> controller ignored")
        else:
            print("[MODE] AUTONOMOUS OFF -> controller ACTIVE")


def handle_control_action(action):
    """
    Handle one command from the website controller
    (forward / backward / left / right / stop) and pass it to the Arduino.
    """

    action = str(action).strip().lower()

    # Toggle sent from the website through the normal control channel.
    if action in ("autonomous_on", "autonomous_off"):
        set_autonomous(action == "autonomous_on")
        return

    if autonomous_mode:
        print(f"[CTRL] Ignored '{action}' (autonomous mode is ON)")
        return

    command = ACTION_TO_COMMAND.get(action)

    if command is None:
        print(f"[CTRL] Unknown action: {action}")
        return

    # Don't allow driving into something in manual mode either.
    if (
        command == "F"
        and ENABLE_MANUAL_OBSTACLE_STOP
        and get_distance_cm() < MANUAL_SAFETY_STOP_CM
    ):
        print("[CTRL] Forward blocked: obstacle too close.")
        send_command("S")
        return

    print(f"[CTRL] Website command: {action} -> {command}")
    send_command(command)


def manual_step():
    """One iteration of the loop while the controller is in charge."""

    refresh_command()

    if (
        ENABLE_MANUAL_OBSTACLE_STOP
        and current_command == "F"
        and get_distance_cm() < MANUAL_SAFETY_STOP_CM
    ):
        print("[CTRL] Obstacle ahead. Stopping.")
        send_command("S")

    time.sleep(0.05)


# ============================================================
# BACKEND LINK (Socket.IO client, runs in a background thread)
# ============================================================

def setup_backend_link():
    global sio

    if socketio is None:
        print(
            "[LINK] python-socketio not installed. Running autonomous only.\n"
            "       Install with: pip install \"python-socketio[client]\" websocket-client"
        )
        return False

    sio = socketio.Client(
        reconnection=True,
        reconnection_delay=2,
        reconnection_delay_max=10
    )

    @sio.event
    def connect():
        print(f"[LINK] Connected to backend {BACKEND_URL}")

    @sio.event
    def disconnect():
        print("[LINK] Disconnected from backend.")
        # Lost the operator while driving manually -> stop, never coast.
        if not autonomous_mode:
            send_command("S")

    @sio.on(MODE_EVENT)
    def on_mode(data):
        if isinstance(data, dict):
            data = data.get("autonomous", True)
        set_autonomous(data)

    @sio.on(CONTROL_EVENT)
    def on_control(data):
        action = data.get("action") if isinstance(data, dict) else data
        handle_control_action(action)

    return True


def backend_worker():
    headers = {"Authorization": f"Bearer {ROBOT_TOKEN}"} if ROBOT_TOKEN else {}

    while running:
        try:
            sio.connect(BACKEND_URL, headers=headers, wait_timeout=5)
            sio.wait()

        except Exception as e:
            print(f"[LINK] Backend unavailable ({e}). Retrying in 5 s...")
            time.sleep(5)


def start_backend_link():
    if setup_backend_link():
        threading.Thread(target=backend_worker, daemon=True).start()


# ============================================================
# SAFE EXIT
# ============================================================

def safe_exit(reason="Program stopped"):
    global running

    if not running:
        return

    running = False

    print("\n")
    print("=" * 60)
    print("[SAFE EXIT] Shutting down rover...")
    print(f"[SAFE EXIT] Reason: {reason}")
    print("=" * 60)

    # FIRST PRIORITY: STOP MOTORS
    try:
        if arduino is not None:
            with serial_lock:
                arduino.write(b"S\n")
                arduino.flush()
            print("[SAFE EXIT] Stop command sent to Arduino.")

    except Exception as e:
        print(f"[SAFE EXIT] Could not send stop command: {e}")

    # DISCONNECT FROM BACKEND
    try:
        if sio is not None and sio.connected:
            sio.disconnect()

    except Exception as e:
        print(f"[SAFE EXIT] Backend disconnect error: {e}")

    # CENTER SERVO
    try:
        if servo is not None:
            servo.value = CENTER_POSITION
            time.sleep(0.2)

    except Exception as e:
        print(f"[SAFE EXIT] Servo cleanup error: {e}")

    # CLOSE SENSOR
    try:
        if sensor is not None:
            sensor.close()

    except Exception as e:
        print(f"[SAFE EXIT] Sensor cleanup error: {e}")

    # CLOSE SERVO
    try:
        if servo is not None:
            servo.close()

    except Exception as e:
        print(f"[SAFE EXIT] Servo close error: {e}")

    # CLOSE SERIAL
    try:
        if arduino is not None:
            arduino.close()
            print("[SAFE EXIT] Arduino serial closed.")

    except Exception as e:
        print(f"[SAFE EXIT] Serial close error: {e}")

    print("[SAFE EXIT] Hardware released.")
    print("[SAFE EXIT] Rover is stopped.")
    print("[SAFE EXIT] Safe to run the program again.")


# ============================================================
# SIGNAL HANDLERS
# ============================================================

def signal_handler(signum, frame):
    safe_exit(f"Received signal {signum}")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ============================================================
# MAIN LOOP
# ============================================================

def run_navigation():

    print()
    print("=" * 60)
    print("🤖 ROVER  (autonomous + website controller)")
    print("=" * 60)

    print(
        f"[AUTO] HC-SR04: GPIO{ULTRASONIC_TRIGGER_PIN} trigger / "
        f"GPIO{ULTRASONIC_ECHO_PIN} echo"
    )
    print(f"[AUTO] Servo: GPIO{SERVO_PIN}")
    print(f"[AUTO] Arduino UART: {ARDUINO_SERIAL_PORT} @ {ARDUINO_BAUD_RATE}")
    print(f"[AUTO] Safe distance: {SAFE_DISTANCE_CM} cm")
    print(f"[LINK] Backend: {BACKEND_URL}")
    print("=" * 60)

    initialize_hardware()

    if servo is not None:
        servo.value = CENTER_POSITION

    time.sleep(1)

    # Start listening for the website toggle / controller.
    start_backend_link()

    print("[AUTO] Navigation started (autonomous mode ON).")

    last_mode = autonomous_mode

    try:

        while running:

            # Mode changed: bring the scanner back to center.
            if autonomous_mode != last_mode:
                last_mode = autonomous_mode

                if servo is not None:
                    servo.value = CENTER_POSITION

            # ---------------- MANUAL (website controller) ----------------
            if not autonomous_mode:
                manual_step()
                continue

            # ---------------- AUTONOMOUS ----------------------------------
            refresh_command()

            distance_cm = get_distance_cm()
            print(f"[AUTO] Distance ahead: {distance_cm:.1f} cm")

            if distance_cm > SAFE_DISTANCE_CM:
                motor_forward()
            else:
                avoid_obstacle()

            time.sleep(LOOP_DELAY)

    except KeyboardInterrupt:
        safe_exit("KeyboardInterrupt")

    except Exception as e:
        print(f"\n[AUTO] Unexpected error: {e}")
        safe_exit("Unexpected program error")
        raise

    finally:
        safe_exit("Navigation loop finished")


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    try:
        run_navigation()

    except Exception as e:
        safe_exit("Fatal program exception")
        print(f"[FATAL] {e}")
        sys.exit(1)