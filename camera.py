from ultralytics import YOLO
import cv2
import urllib.request
import numpy as np
import os
import time

# ---------- CONFIG ----------
MODEL_PATH = "blacknwhite.pt"
SNAPSHOT_URL = "http://10.247.208.239/capture"
CROPS_FOLDER = r"C:\Users\quant\OneDrive\Desktop\ROVER-FEATURE\results\face_crops"
SNAPSHOT_DELAY = 0.1  # seconds between frames (lower = faster, but may overload ESP32)
# ----------------------------

model = YOLO(MODEL_PATH)
os.makedirs(CROPS_FOLDER, exist_ok=True)

total_faces = 0

def process_frame(frame):
    global total_faces
    results = model.predict(source=frame, conf=0.5, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue
        crop_path = os.path.join(CROPS_FOLDER, f"{total_faces}_face.jpg")
        cv2.imwrite(crop_path, face_crop)
        total_faces += 1
        print(f"Saved: {crop_path}")

print(f"Starting snapshot capture from {SNAPSHOT_URL}")
print("Press Ctrl+C to stop.\n")

frame_count = 0

try:
    while True:
        try:
            req = urllib.request.urlopen(SNAPSHOT_URL, timeout=5)
            data = req.read()
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)

            if frame is None:
                print("Warning: got empty frame, skipping...")
                continue

            frame_count += 1
            print(f"Frame {frame_count} received ({frame.shape[1]}x{frame.shape[0]})", end=" | ")
            process_frame(frame)

            time.sleep(SNAPSHOT_DELAY)

        except urllib.error.URLError as e:
            print(f"Network error: {e} — retrying in 2s...")
            time.sleep(2)

except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    print(f"\nDone. {total_faces} face crop(s) saved from {frame_count} frames.")