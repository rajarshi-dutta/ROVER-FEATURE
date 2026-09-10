"""
Face Recognition Module — Direct In-Memory Matching + Unknown-Face Dedup
==========================================================================
camera.py calls identify_face(face_crop) directly for every detected face,
in memory, before anything is written to disk.

This module now owns the full decision for unknown faces too:
  1. Compare the face against the known-faces DB (enrolled photos).
  2. If it doesn't match anyone known, compare it against faces already
     saved in the unknown-faces folder (by embedding similarity, not a
     perceptual hash — much more robust to angle/lighting changes).
  3. If a similar unknown face already exists there, skip saving (it's
     the same unrecognized person as before).
  4. Otherwise, save the crop into the unknown-faces folder and remember
     its embedding so future frames of the same person get skipped too.

Usage:
    import facerecog

    facerecog.init_recognition()                  # once, at startup
    result = facerecog.identify_face(face_crop)    # per detected face

    result["status"] is one of:
        "MATCHED"       -> known person, already discarded (nothing saved)
        "NEW_UNKNOWN"   -> unknown face, newly saved (see result["saved_path"])
        "DUPLICATE"     -> unknown face, but already saved before (skipped)
        "NO_FACE"       -> crop was too poor to get an embedding from
        "ERROR"         -> empty/invalid crop
"""

import cv2
import numpy as np
import os
import threading
from pathlib import Path

try:
    from insightface.app import FaceAnalysis
except ImportError:
    raise ImportError("Run: pip install insightface onnxruntime opencv-python numpy")


# ============================================================================
# CONFIGURATION
# ============================================================================

KNOWN_FACES_FOLDER = r"C:\Users\Rajarshi\OneDrive\Desktop\Robo dog\known_faces"
UNKNOWN_FACES_FOLDER = r"C:\Users\Rajarshi\OneDrive\Desktop\Robo dog\results\unknown_faces"

MATCH_THRESHOLD = 0.30           # known-face match threshold
UNKNOWN_DEDUP_THRESHOLD = 0.35   # "same unrecognized person" threshold — stricter
                                  # than MATCH_THRESHOLD since a false dedup-skip
                                  # just means one fewer photo of the same stranger.
MIN_FACE_SIZE = 120

# ============================================================================

# Global state
_APP = None
_MODEL_LOCK = threading.Lock()
_known_db = {}
_db_lock = threading.Lock()
_ready = False

# Unknown-faces state: filename -> embedding
_unknown_db = {}
_unknown_lock = threading.Lock()
_unknown_counter = 0


def init_recognition():
    """
    Load the InsightFace model, build the known-faces database, and load
    any unknown faces already saved on disk (e.g. from a previous run) so
    dedup keeps working across restarts.
    Call this once at startup, before the camera loop begins.
    """
    global _ready
    _init_model()
    _build_known_db()
    _load_existing_unknowns()
    _ready = True
    print("[FACEREC] Ready for direct frame-by-frame matching.\n")


def is_ready():
    return _ready


def _init_model():
    global _APP
    if _APP is None:
        print("[FACEREC] Loading InsightFace model…")
        _APP = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _APP.prepare(ctx_id=0, det_size=(320, 320))
        print("[FACEREC] Model ready.\n")


def _preprocess(image: np.ndarray) -> np.ndarray:
    """Prepare a face crop for InsightFace — upscale if tiny."""
    h, w = image.shape[:2]
    short_side = min(h, w)

    if short_side < MIN_FACE_SIZE:
        scale = MIN_FACE_SIZE / short_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    return image


def _get_embedding(image: np.ndarray):
    """Extract ArcFace embedding from a face crop (thread-safe)."""
    global _APP

    img = _preprocess(image)

    with _MODEL_LOCK:
        faces = _APP.get(img)

        if not faces:
            h, w = img.shape[:2]
            pad = max(h, w) // 4
            padded = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                       cv2.BORDER_CONSTANT, value=(128, 128, 128))
            faces = _APP.get(padded)

    if not faces:
        return None

    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return best.normed_embedding


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _build_known_db():
    """Build known faces database once at startup."""
    global _known_db

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    folder = Path(KNOWN_FACES_FOLDER)

    if not folder.exists():
        print(f"[FACEREC] Known faces folder not found: {KNOWN_FACES_FOLDER}")
        return

    images = [p for p in folder.iterdir() if p.suffix.lower() in exts]
    if not images:
        print(f"[FACEREC] No images found in: {KNOWN_FACES_FOLDER}")
        return

    print(f"[FACEREC] Building known-face database from {len(images)} image(s)…")
    db = {}

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        print(f"[FACEREC]   Reading: {img_path.name}  ({w}×{h} px)")

        emb = _get_embedding(img)
        if emb is None:
            print(f"[FACEREC]   [!] No face detected in: {img_path.name}")
            continue

        name = img_path.stem.replace("_", " ").replace("-", " ").title()
        db[name] = emb
        print(f"[FACEREC]   ✓ Enrolled: {name}")

    with _db_lock:
        _known_db = db

    if db:
        print(f"[FACEREC] Database ready — {len(db)} person(s): {', '.join(db)}\n")
    else:
        print(f"[FACEREC] Warning: No faces enrolled in database.\n")


def _load_existing_unknowns():
    """
    On startup, embed any unknown faces already saved on disk so dedup
    keeps working across restarts instead of re-saving the same stranger.
    """
    global _unknown_db, _unknown_counter

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    folder = Path(UNKNOWN_FACES_FOLDER)
    folder.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
    if not images:
        _unknown_counter = 0
        return

    print(f"[FACEREC] Loading {len(images)} existing unknown face(s) for dedup…")
    db = {}
    max_index = -1

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        emb = _get_embedding(img)
        if emb is None:
            continue

        db[img_path.name] = emb

        # Filenames look like "<n>_unknown.jpg" — recover n to keep
        # counting up instead of overwriting existing files.
        try:
            idx = int(img_path.stem.split("_")[0])
            max_index = max(max_index, idx)
        except ValueError:
            pass

    with _unknown_lock:
        _unknown_db = db
        _unknown_counter = max_index + 1

    print(f"[FACEREC] Unknown-face dedup primed with {len(db)} existing face(s).\n")


def _find_similar_unknown(emb):
    """Return the filename of the closest already-saved unknown face if it's
    similar enough to count as the same person, else None."""
    with _unknown_lock:
        best_filename = None
        best_similarity = -1.0
        for filename, saved_emb in _unknown_db.items():
            sim = _cosine(emb, saved_emb)
            if sim > best_similarity:
                best_similarity = sim
                best_filename = filename

        if best_filename is not None and best_similarity >= UNKNOWN_DEDUP_THRESHOLD:
            return best_filename, best_similarity
        return None, best_similarity


def _save_unknown(face_crop, emb):
    """Save a newly-seen unknown face to disk and remember its embedding."""
    global _unknown_counter

    os.makedirs(UNKNOWN_FACES_FOLDER, exist_ok=True)

    with _unknown_lock:
        filename = f"{_unknown_counter}_unknown.jpg"
        path = os.path.join(UNKNOWN_FACES_FOLDER, filename)
        cv2.imwrite(path, face_crop)
        _unknown_db[filename] = emb
        _unknown_counter += 1

    return path


def identify_face(face_crop: np.ndarray) -> dict:
    """
    Identify a single face crop (numpy BGR array) directly from memory,
    and fully resolve what should happen to it:
      - a known person -> nothing is saved
      - a new unknown face -> saved to UNKNOWN_FACES_FOLDER
      - a repeat unknown face (already in that folder) -> skipped

    Returns a dict:
        status: "MATCHED" | "NEW_UNKNOWN" | "DUPLICATE" | "NO_FACE" | "ERROR"
        matched_person: str (for MATCHED)
        similarity: float (best match similarity, known or unknown context)
        saved_path: str (present only for NEW_UNKNOWN)
        top_candidates: list[dict] (known-face comparisons, if any ran)
    """
    result = {
        "status": "ERROR",
        "matched_person": "UNKNOWN",
        "similarity": 0.0,
        "saved_path": None,
        "top_candidates": [],
        "error": "",
    }

    if face_crop is None or face_crop.size == 0:
        result["error"] = "Empty face crop"
        return result

    emb = _get_embedding(face_crop)
    if emb is None:
        result["status"] = "NO_FACE"
        result["error"] = "No face detected in crop"
        return result

    # --- Step 1: check against known people ---
    with _db_lock:
        known_items = list(_known_db.items())

    if known_items:
        comparisons = [
            {"person_name": name, "similarity": round(_cosine(emb, known_emb), 4)}
            for name, known_emb in known_items
        ]
        comparisons_sorted = sorted(comparisons, key=lambda r: r["similarity"], reverse=True)
        best = comparisons_sorted[0]
        result["top_candidates"] = comparisons_sorted[:3]

        if best["similarity"] >= MATCH_THRESHOLD:
            result["status"] = "MATCHED"
            result["matched_person"] = best["person_name"]
            result["similarity"] = best["similarity"]
            return result

    # --- Step 2: not a known person — check the unknown-faces folder ---
    dup_filename, dup_similarity = _find_similar_unknown(emb)

    if dup_filename is not None:
        result["status"] = "DUPLICATE"
        result["matched_person"] = dup_filename
        result["similarity"] = dup_similarity
        return result

    # --- Step 3: genuinely new unknown face — save it ---
    saved_path = _save_unknown(face_crop, emb)
    result["status"] = "NEW_UNKNOWN"
    result["saved_path"] = saved_path
    result["similarity"] = dup_similarity if dup_similarity > 0 else 0.0
    return result


def get_known_faces():
    """Get list of enrolled known faces."""
    with _db_lock:
        return list(_known_db.keys())


def get_unknown_face_count():
    """Get the number of distinct unknown faces saved so far."""
    with _unknown_lock:
        return len(_unknown_db)