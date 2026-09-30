"""
Quick Pi -> Arduino test.
  GPIO wires:   python serial_test.py
  USB cable:    python serial_test.py /dev/ttyUSB0     (or /dev/ttyACM0)
Lift the rover off the ground first!  Press Ctrl+C to abort.
"""
import sys
import time
import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/serial0"
BAUD = 115200          # must match Serial.begin() in the Arduino sketch

print(f"Opening {PORT} at {BAUD} baud...")
ser = serial.Serial(PORT, BAUD, timeout=0.1, write_timeout=1)

print("Port open. Waiting 2 s for the Nano to boot...")
time.sleep(2.0)                   # a Nano resets when the port opens
ser.reset_input_buffer()

try:
    print("Sending F (forward) for 2 seconds...")
    end = time.time() + 2
    n = 0
    while time.time() < end:
        n += 1
        try:
            ser.write(b"F\n")
            print(f"  sent F #{n}")
        except serial.SerialTimeoutException:
            print("  WRITE TIMED OUT - the Nano is not accepting data")
            break
        time.sleep(0.25)          # keep the Arduino watchdog fed
        reply = ser.read(ser.in_waiting or 1)
        if reply:
            print("  Arduino says:", reply.decode(errors="replace").strip())

finally:
    try:
        ser.write(b"S\n")
        print("Sent S (stop).")
    except Exception as e:
        print("Could not send S:", e)
    ser.close()
    print("Done.")