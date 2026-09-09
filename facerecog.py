"""
Face Matching System — InsightFace ArcFace
==========================================
IMPORTANT: Point this at face_crops (raw) NOT face_crops_enlarged.
Enhancement and upscaling help human viewing, NOT ArcFace recognition.
ArcFace does its own internal normalization to 112×112.

Install:
    pip install insightface onnxruntime opencv-python numpy
"""

import cv2
import numpy as np
import os
from pathlib import Path

try:
    from insightface.app import FaceAnalysis
except ImportError:
    raise ImportError("Run: pip install insightface onnxruntime opencv-python numpy")


# ============================================================================
# CONFIGURATION  ← only edit this section
# ============================================================================

# !! Use raw crops, NOT enlarged/enhanced crops !!
DETECTED_FACES_FOLDER = r"C:\Users\Rajarshi\OneDrive\Desktop\Robo dog\results\face_crops"

KNOWN_FACES_FOLDER = r"C:\Users\Rajarshi\OneDrive\Desktop\Robo dog\known_faces"

# Threshold: 0.30 = lenient, 0.40 = normal, 0.50 = strict
MATCH_THRESHOLD = 0.30

# If crops are tiny (< 80px), upscale before sending to InsightFace
# This is different from your pipeline upscaling — this is purely for detection
MIN_FACE_SIZE = 80   # pixels (shorter side)


# ============================================================================
# MODEL  — loaded once
# ============================================================================

print("Loading InsightFace model…")
_APP = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
_APP.prepare(ctx_id=0, det_size=(320, 320))   # smaller det_size = finds smaller faces
print("Model ready.\n")


# ============================================================================
# HELPERS
# ============================================================================

def _preprocess(image: np.ndarray) -> np.ndarray:
    """
    Prepare a face crop for InsightFace:
    - If the crop is tiny, upscale it so the detector can find the face.
    - Do NOT apply enhancement — it distorts the embedding.
    """
    h, w = image.shape[:2]
    short_side = min(h, w)

    if short_side < MIN_FACE_SIZE:
        scale  = MIN_FACE_SIZE / short_side
        new_w  = int(w * scale)
        new_h  = int(h * scale)
        image  = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    return image


def _get_embedding(image: np.ndarray, path_hint: str = "") -> np.ndarray | None:
    """
    Extract ArcFace embedding from a face crop.
    Falls back to a larger det_size if the first attempt fails.
    """
    img = _preprocess(image)
    faces = _APP.get(img)

    # If nothing found, try a larger detection window
    if not faces:
        h, w  = img.shape[:2]
        # pad the image so the detector has more context
        pad   = max(h, w) // 4
        padded = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                    cv2.BORDER_CONSTANT, value=(128, 128, 128))
        faces = _APP.get(padded)

    if not faces:
        return None

    best = max(faces,
               key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return best.normed_embedding


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _load_image(path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        print(f"  [!] Cannot read: {path}")
    return img


# ============================================================================
# BUILD KNOWN DATABASE
# ============================================================================

def build_known_db(known_folder: str) -> dict[str, np.ndarray]:
    exts   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    folder = Path(known_folder)

    if not folder.exists():
        raise FileNotFoundError(f"Known faces folder not found: {known_folder}")

    images = [p for p in folder.iterdir() if p.suffix.lower() in exts]
    if not images:
        raise ValueError(f"No images found in: {known_folder}")

    print(f"Building known-face database from {len(images)} image(s)…")
    db: dict[str, np.ndarray] = {}

    for img_path in images:
        img = _load_image(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        print(f"  Reading: {img_path.name}  ({w}×{h} px)")

        emb = _get_embedding(img, str(img_path))
        if emb is None:
            print(f"  [!] No face detected in known image: {img_path.name}")
            print(f"      → Make sure the photo is a clear, well-lit, front-facing shot")
            continue

        name    = img_path.stem.replace("_", " ").replace("-", " ").title()
        db[name] = emb
        print(f"  ✓ Enrolled: {name}")

    if not db:
        raise ValueError("No faces could be enrolled. Check your known_faces images.")

    print(f"\nDatabase ready — {len(db)} person(s): {', '.join(db)}\n")
    return db


# ============================================================================
# MATCH ALL FACES
# ============================================================================

def match_all_faces(source_folder: str, known_folder: str,
                    threshold: float = 0.30) -> list[dict]:

    print(f"\n{'='*70}")
    print("Face Matching  —  InsightFace ArcFace")
    print(f"{'='*70}")
    print(f"  Detected faces : {source_folder}")
    print(f"  Known faces    : {known_folder}")
    print(f"  Threshold      : {threshold:.2f}\n")

    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"Source folder not found: {source_folder}")

    db = build_known_db(known_folder)

    exts  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(f for f in os.listdir(source_folder)
                   if Path(f).suffix.lower() in exts)

    print(f"Found {len(files)} detected face(s) to match\n")
    all_matches: list[dict] = []

    for idx, filename in enumerate(files, 1):
        path = os.path.join(source_folder, filename)
        img  = _load_image(path)

        print(f"{idx}. {filename}")

        if img is None:
            print("   ✗ Could not read image\n")
            continue

        h, w = img.shape[:2]
        print(f"   Size: {w}×{h} px", end="")

        emb = _get_embedding(img, path)

        if emb is None:
            print("  →  ✗ No face detected")
            print(f"      Tip: crop is {w}×{h} px — "
                  + ("very small, try lowering MIN_FACE_SIZE or increasing UPSCALE_FACTOR in pipeline"
                     if min(w, h) < 60 else
                     "face may be blurry, occluded, or at a sharp angle"))
            print()
            all_matches.append({"detected_image": filename,
                                 "matched_person": "NO_FACE",
                                 "similarity": 0.0, "status": "NO_FACE"})
            continue

        # Compare against all known faces
        results = sorted(
            [{"person_name": name,
              "similarity":  round(_cosine(emb, known_emb), 4)}
             for name, known_emb in db.items()],
            key=lambda r: r["similarity"], reverse=True
        )
        best = results[0]

        if best["similarity"] >= threshold:
            print(f"  →  ✓ MATCHED: {best['person_name']}  "
                  f"({best['similarity']:.4f} / {best['similarity']*100:.1f}%)")
            status = "MATCHED"
        else:
            print(f"  →  ✗ NO MATCH  "
                  f"(closest: {best['person_name']} @ {best['similarity']:.4f})")
            status = "NOT_MATCHED"

        print("   Top candidates:")
        for rank, r in enumerate(results[:3], 1):
            flag = "✓" if r["similarity"] >= threshold else " "
            print(f"     {flag}{rank}. {r['person_name']:30s}  {r['similarity']:.4f}")
        print()

        all_matches.append({
            "detected_image": filename,
            "matched_person": best["person_name"] if status == "MATCHED" else "UNKNOWN",
            "similarity":     best["similarity"],
            "status":         status,
        })

    # Summary
    matched   = sum(1 for m in all_matches if m["status"] == "MATCHED")
    unmatched = sum(1 for m in all_matches if m["status"] == "NOT_MATCHED")
    no_face   = sum(1 for m in all_matches if m["status"] == "NO_FACE")

    print(f"{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Total processed : {len(all_matches)}")
    print(f"  ✓ Matched       : {matched}")
    print(f"  ✗ Not matched   : {unmatched}")
    print(f"  ✗ No face found : {no_face}\n")

    col = "{:<45} {:<20} {:<10}"
    print(col.format("Detected Image", "Matched Person", "Similarity"))
    print("-" * 70)
    for m in all_matches:
        sim = f"{m['similarity']:.4f}" if m["similarity"] else "N/A"
        print(col.format(m["detected_image"][:44], m["matched_person"], sim))
    print(f"{'='*70}\n")

    return all_matches


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    match_all_faces(
        source_folder=DETECTED_FACES_FOLDER,
        known_folder=KNOWN_FACES_FOLDER,
        threshold=MATCH_THRESHOLD,
    )