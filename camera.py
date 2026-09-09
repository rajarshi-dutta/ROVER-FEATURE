from ultralytics import YOLO
import cv2
import os

# ---------- CONFIG ----------
MODEL_PATH = "blacknwhite.pt"
SOURCE_FOLDER = r"C:\Users\Rajarshi\OneDrive\Desktop\Robo dog\test image foldere"
CROPS_FOLDER = "C:/Users/Rajarshi/OneDrive/Desktop/Robo dog/results/face_crops"
# -----------------------------

model = YOLO(MODEL_PATH)
os.makedirs(CROPS_FOLDER, exist_ok=True)

total_faces = 0

for filename in os.listdir(SOURCE_FOLDER):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(SOURCE_FOLDER, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not read {filename}, skipping")
        continue

    results = model.predict(source=image, verbose=False)
    base_name = os.path.splitext(filename)[0]

    boxes = results[0].boxes.xyxy.cpu().numpy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        face_crop = image[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        crop_path = os.path.join(CROPS_FOLDER, f"{base_name}_face_{i}.jpg")
        cv2.imwrite(crop_path, face_crop)
        total_faces += 1

    print(f"Processed {filename}: {len(boxes)} face(s) found")

print(f"\nDone. {total_faces} face crop(s) saved to {CROPS_FOLDER}")