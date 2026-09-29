import time
import signal
import sys
import serial

from gpiozero import DistanceSensor, Servo


# ============================================================
# SAFETY CONFIGURATION
# ============================================================

# Distance below which the rover considers the path dangerous.
SAFE_DISTANCE_CM = 30.0

# Maximum distance used by HC-SR04.
MAX_SENSOR_DISTANCE_M = 2.0

# Time allowed for servo to settle after moving.
SERVO_SETTLE_TIME = 0.4

# Time between normal distance checks.
LOOP_DELAY = 0.1

# How long the Arduino is allowed to keep the last
# movement command before its own watchdog stops the motors.
# This value must be larger than the longest movement delay below.
ARDUINO_WATCHDOG_SECONDS = 2.0

# How often we resend the current movement command.
# This keeps the Arduino watchdog alive while moving.
COMMAND_REFRESH_INTERVAL = 0.25


# ============================================================
# RASPBERRY PI GPIO CONFIGURATION
# ============================================================

# HC-SR04
ULTRASONIC_TRIGGER_PIN = 23
ULTRASONIC_ECHO_PIN = 24

# Servo
SERVO_PIN = 18

# Raspberry Pi UART
#
# /dev/serial0 is recommended because it points to the
# primary UART regardless of which physical UART is assigned.
#
# If this does not work, check:
#     ls -l /dev/serial0
#
ARDUINO_SERIAL_PORT = "/dev/serial0"
ARDUINO_BAUD_RATE = 9600


# ============================================================
# SERVO POSITIONS
# ============================================================

# gpiozero Servo:
# -1 = one extreme
#  0 = center
# +1 = other extreme

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

running = True

current_command = "S"
last_command_time = 0.0


# ============================================================
# SERIAL COMMUNICATION
# ============================================================

def connect_to_arduino():
    """
    Open the UART connection to the Arduino Nano.

    Returns:
        True  -> connection successful
        False -> connection failed
    """

    global arduino

    try:
        print(
            f"[SERIAL] Connecting to Arduino on "
            f"{ARDUINO_SERIAL_PORT}..."
        )

        arduino = serial.Serial(
            port=ARDUINO_SERIAL_PORT,
            baudrate=ARDUINO_BAUD_RATE,
            timeout=1
        )

        time.sleep(0.2)

        print("[SERIAL] Arduino connected.")
        return True

    except Exception as e:
        print(f"[SERIAL] Could not connect to Arduino: {e}")
        arduino = None
        return False


def send_command(command):
    """
    Send a movement command to the Arduino.

    Commands:
        F = Forward
        B = Backward
        L = Left
        R = Right
        S = Stop
    """

    global current_command
    global last_command_time

    if command not in ("F", "B", "L", "R", "S"):
        print(f"[SERIAL] Invalid command: {command}")
        return False

    current_command = command
    last_command_time = time.monotonic()

    if arduino is None:
        print(
            f"[SERIAL] Arduino unavailable. "
            f"Command {command} not sent."
        )
        return False

    try:
        message = command + "\n"
        arduino.write(message.encode("ascii"))
        arduino.flush()

        print(f"[SERIAL] → Arduino: {command}")

        return True

    except Exception as e:
        print(f"[SERIAL] Send error: {e}")

        # Don't allow serial failure to crash the program.
        return False


def refresh_command():
    """
    Periodically resend the current command.

    This prevents the Arduino watchdog from stopping the rover
    while the Pi is intentionally moving.
    """

    global last_command_time

    if current_command == "S":
        return

    now = time.monotonic()

    if now - last_command_time >= COMMAND_REFRESH_INTERVAL:
        send_command(current_command)


# ============================================================
# MOTOR COMMAND FUNCTIONS
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
    """
    Initialize Raspberry Pi hardware and Arduino connection.
    """

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
    Read the HC-SR04 distance.

    Returns:
        Distance in centimeters.

    A sensor failure is treated as an unsafe condition.
    """

    if sensor is None:
        return 0.0

    try:
        distance_cm = sensor.distance * 100.0

        if distance_cm < 0:
            return 0.0

        return distance_cm

    except Exception as e:
        print(f"[AUTO] Ultrasonic sensor error: {e}")

        # SAFETY:
        # If we cannot determine the distance,
        # assume the path is blocked.
        return 0.0


# ============================================================
# SERVO SCANNING
# ============================================================

def scan_direction(position, name):
    """
    Move the ultrasonic sensor to a direction,
    allow the servo to settle, and measure distance.
    """

    if servo is None:
        return 0.0

    print(f"[SCAN] Looking {name}...")

    try:
        servo.value = position

        time.sleep(SERVO_SETTLE_TIME)

        distance = get_distance_cm()

        print(
            f"[SCAN] {name}: "
            f"{distance:.1f} cm"
        )

        return distance

    except Exception as e:
        print(
            f"[SCAN] Servo/sensor error while "
            f"looking {name}: {e}"
        )

        return 0.0


def scan_environment():
    """
    Scan LEFT, CENTER and RIGHT.

    Returns:
        left_distance,
        center_distance,
        right_distance
    """

    print("\n[SCAN] Scanning environment...")

    left_distance = scan_direction(
        LEFT_POSITION,
        "LEFT"
    )

    center_distance = scan_direction(
        CENTER_POSITION,
        "CENTER"
    )

    right_distance = scan_direction(
        RIGHT_POSITION,
        "RIGHT"
    )

    # Return sensor to center.
    if servo is not None:
        servo.value = CENTER_POSITION

    time.sleep(0.2)

    print(
        f"[SCAN] Result -> "
        f"L: {left_distance:.1f} cm | "
        f"C: {center_distance:.1f} cm | "
        f"R: {right_distance:.1f} cm"
    )

    return (
        left_distance,
        center_distance,
        right_distance
    )


# ============================================================
# DIRECTION DECISION
# ============================================================

def choose_direction(left, center, right):
    """
    Decide which direction provides the most space.

    Returns:
        forward
        left
        right
        backward
    """

    print("[AUTO] Choosing navigation direction...")

    # Center is safe.
    if center > SAFE_DISTANCE_CM:
        print("[AUTO] Center is clear -> FORWARD")
        return "forward"

    print("[AUTO] Center blocked!")

    # Left is safe and better than right.
    if (
        left > SAFE_DISTANCE_CM
        and left > right
    ):
        print(
            f"[AUTO] Left is clearer "
            f"({left:.1f} cm) -> LEFT"
        )

        return "left"

    # Right is safe and better than left.
    if (
        right > SAFE_DISTANCE_CM
        and right > left
    ):
        print(
            f"[AUTO] Right is clearer "
            f"({right:.1f} cm) -> RIGHT"
        )

        return "right"

    # Neither side is safely clear.
    # Choose the side with more space.
    if left > right:
        print(
            "[AUTO] Both sides restricted. "
            "Left has more space -> LEFT"
        )

        return "left"

    if right > left:
        print(
            "[AUTO] Both sides restricted. "
            "Right has more space -> RIGHT"
        )

        return "right"

    # Equal or zero distances.
    print(
        "[AUTO] No safe direction found -> BACKWARD"
    )

    return "backward"


# ============================================================
# OBSTACLE AVOIDANCE
# ============================================================

def avoid_obstacle():
    """
    Stop, scan surroundings and select a direction.
    """

    print("\n⚠️ [AUTO] Obstacle detected!")

    # Immediately stop.
    motor_stop()

    time.sleep(0.2)

    left, center, right = scan_environment()

    direction = choose_direction(
        left,
        center,
        right
    )

    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------

    if direction == "forward":

        motor_forward()

        start_time = time.monotonic()

        while (
            time.monotonic() - start_time
            < FORWARD_AFTER_SCAN_TIME
        ):
            refresh_command()
            time.sleep(0.05)

        motor_stop()

    # --------------------------------------------------------
    # LEFT
    # --------------------------------------------------------

    elif direction == "left":

        motor_turn_left()

        start_time = time.monotonic()

        while (
            time.monotonic() - start_time
            < TURN_LEFT_TIME
        ):
            refresh_command()
            time.sleep(0.05)

        motor_stop()

    # --------------------------------------------------------
    # RIGHT
    # --------------------------------------------------------

    elif direction == "right":

        motor_turn_right()

        start_time = time.monotonic()

        while (
            time.monotonic() - start_time
            < TURN_RIGHT_TIME
        ):
            refresh_command()
            time.sleep(0.05)

        motor_stop()

    # --------------------------------------------------------
    # BACKWARD
    # --------------------------------------------------------

    elif direction == "backward":

        motor_backward()

        start_time = time.monotonic()

        while (
            time.monotonic() - start_time
            < BACKWARD_TIME
        ):
            refresh_command()
            time.sleep(0.05)

        motor_stop()


# ============================================================
# SAFE EXIT
# ============================================================

def safe_exit(reason="Program stopped"):
    """
    Safely shut down the rover.

    This function is intentionally designed to be safe to call
    multiple times.
    """

    global running

    if not running:
        return

    running = False

    print("\n")
    print("=" * 60)
    print("[SAFE EXIT] Shutting down rover...")
    print(f"[SAFE EXIT] Reason: {reason}")
    print("=" * 60)

    # --------------------------------------------------------
    # FIRST PRIORITY: STOP MOTORS
    # --------------------------------------------------------

    try:
        if arduino is not None:
            arduino.write(b"S\n")
            arduino.flush()
            print("[SAFE EXIT] Stop command sent to Arduino.")

    except Exception as e:
        print(
            f"[SAFE EXIT] Could not send stop command: {e}"
        )

    # --------------------------------------------------------
    # CENTER SERVO
    # --------------------------------------------------------

    try:
        if servo is not None:
            servo.value = CENTER_POSITION
            time.sleep(0.2)

    except Exception as e:
        print(
            f"[SAFE EXIT] Servo cleanup error: {e}"
        )

    # --------------------------------------------------------
    # CLOSE SENSOR
    # --------------------------------------------------------

    try:
        if sensor is not None:
            sensor.close()

    except Exception as e:
        print(
            f"[SAFE EXIT] Sensor cleanup error: {e}"
        )

    # --------------------------------------------------------
    # CLOSE SERVO
    # --------------------------------------------------------

    try:
        if servo is not None:
            servo.close()

    except Exception as e:
        print(
            f"[SAFE EXIT] Servo close error: {e}"
        )

    # --------------------------------------------------------
    # CLOSE SERIAL
    # --------------------------------------------------------

    try:
        if arduino is not None:
            arduino.close()
            print("[SAFE EXIT] Arduino serial closed.")

    except Exception as e:
        print(
            f"[SAFE EXIT] Serial close error: {e}"
        )

    print("[SAFE EXIT] Hardware released.")
    print("[SAFE EXIT] Rover is stopped.")
    print("[SAFE EXIT] Safe to run the program again.")


# ============================================================
# SIGNAL HANDLERS
# ============================================================

def signal_handler(signum, frame):
    """
    Handle Ctrl+C and termination signals.
    """

    safe_exit(
        f"Received signal {signum}"
    )

    sys.exit(0)


signal.signal(
    signal.SIGINT,
    signal_handler
)

signal.signal(
    signal.SIGTERM,
    signal_handler
)


# ============================================================
# MAIN AUTONOMOUS NAVIGATION
# ============================================================

def run_autonomous_navigation():

    print()
    print("=" * 60)
    print("🤖 AUTONOMOUS ROVER")
    print("=" * 60)

    print(
        f"[AUTO] HC-SR04: "
        f"GPIO{ULTRASONIC_TRIGGER_PIN} trigger / "
        f"GPIO{ULTRASONIC_ECHO_PIN} echo"
    )

    print(
        f"[AUTO] Servo: GPIO{SERVO_PIN}"
    )

    print(
        f"[AUTO] Arduino UART: "
        f"{ARDUINO_SERIAL_PORT} @ "
        f"{ARDUINO_BAUD_RATE}"
    )

    print(
        f"[AUTO] Safe distance: "
        f"{SAFE_DISTANCE_CM} cm"
    )

    print("=" * 60)

    initialize_hardware()

    # Make absolutely sure the sensor starts centered.
    if servo is not None:
        servo.value = CENTER_POSITION

    time.sleep(1)

    print("[AUTO] Navigation started.")

    try:

        while running:

            # Keep Arduino informed while moving.
            refresh_command()

            distance_cm = get_distance_cm()

            print(
                f"[AUTO] Distance ahead: "
                f"{distance_cm:.1f} cm"
            )

            # ------------------------------------------------
            # PATH CLEAR
            # ------------------------------------------------

            if distance_cm > SAFE_DISTANCE_CM:

                motor_forward()

                time.sleep(LOOP_DELAY)

            # ------------------------------------------------
            # OBSTACLE
            # ------------------------------------------------

            else:

                avoid_obstacle()

                time.sleep(LOOP_DELAY)

    except KeyboardInterrupt:

        safe_exit("KeyboardInterrupt")

    except Exception as e:

        print(
            f"\n[AUTO] Unexpected error: {e}"
        )

        safe_exit(
            "Unexpected program error"
        )

        raise

    finally:

        safe_exit(
            "Navigation loop finished"
        )


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    try:

        run_autonomous_navigation()

    except Exception as e:

        # Last line of defense.
        safe_exit(
            "Fatal program exception"
        )

        print(
            f"[FATAL] {e}"
        )

        sys.exit(1)
