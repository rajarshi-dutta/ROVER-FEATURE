"""
Face Recognition Module
=======================

Known-face matching + unknown-face deduplication.

When a genuinely new unknown person is detected:

1. Save the image locally.
2. Send the saved image to the website backend.
3. Website backend creates the security event.
4. Website backend sends the email.
"""

import cv2
import numpy as np
import os
import threading
import requests

from pathlib import Path
from dotenv import load_dotenv

try:
    from insightface.app import FaceAnalysis
except ImportError:
    raise ImportError(
        "Run: pip install insightface onnxruntime opencv-python numpy"
    )


# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()

KNOWN_FACES_FOLDER = os.getenv("kface")
UNKNOWN_FACES_FOLDER = os.getenv("uface")

# --------------------------------------------------------------------------
# Website intrusion reporting
# --------------------------------------------------------------------------

# Example:
# http://192.168.1.100:5050/api/robot/intruder/report
INTRUSION_REPORT_URL = os.getenv("INTRUSION_REPORT_URL", "").strip()

# Must be the same value as INTRUSION_PUSH_TOKEN on the website server.
INTRUSION_PUSH_TOKEN = os.getenv("INTRUSION_PUSH_TOKEN", "").strip()

INTRUSION_REPORT_TIMEOUT = float(
    os.getenv("INTRUSION_REPORT_TIMEOUT", "10")
)

# --------------------------------------------------------------------------

MATCH_THRESHOLD = 0.30

UNKNOWN_DEDUP_THRESHOLD = 0.35

MIN_FACE_SIZE = 120


# ============================================================================
# GLOBAL STATE
# ============================================================================

_APP = None

_MODEL_LOCK = threading.Lock()

_known_db = {}
_db_lock = threading.Lock()

_ready = False


# Unknown faces:
# filename -> embedding
_unknown_db = {}

_unknown_lock = threading.Lock()

_unknown_counter = 0


# ============================================================================
# INITIALIZATION
# ============================================================================

def init_recognition():
    """
    Load InsightFace, build known-face database,
    and load previously saved unknown faces.
    """

    global _ready

    _init_model()

    _build_known_db()

    _load_existing_unknowns()

    _ready = True

    print("[FACEREC] Ready for direct frame-by-frame matching.\n")


def is_ready():
    return _ready


# ============================================================================
# INSIGHTFACE MODEL
# ============================================================================

def _init_model():
    global _APP

    if _APP is None:

        print("[FACEREC] Loading InsightFace model...")

        _APP = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        _APP.prepare(
            ctx_id=0,
            det_size=(320, 320)
        )

        print("[FACEREC] Model ready.\n")


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def _preprocess(image: np.ndarray) -> np.ndarray:

    h, w = image.shape[:2]

    short_side = min(h, w)

    if short_side < MIN_FACE_SIZE:

        scale = MIN_FACE_SIZE / short_side

        new_w = int(w * scale)
        new_h = int(h * scale)

        image = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_LANCZOS4
        )

    return image


# ============================================================================
# FACE EMBEDDING
# ============================================================================

def _get_embedding(image: np.ndarray):

    global _APP

    img = _preprocess(image)

    with _MODEL_LOCK:

        # First attempt
        faces = _APP.get(img)

        # Fallback with padding
        if not faces:

            h, w = img.shape[:2]

            pad = max(h, w) // 2

            padded = cv2.copyMakeBorder(
                img,
                pad,
                pad,
                pad,
                pad,
                cv2.BORDER_CONSTANT,
                value=(128, 128, 128)
            )

            faces = _APP.get(padded)

    if not faces:
        return None

    # Select largest detected face
    best = max(
        faces,
        key=lambda f:
        (f.bbox[2] - f.bbox[0]) *
        (f.bbox[3] - f.bbox[1])
    )

    return best.normed_embedding


# ============================================================================
# COSINE SIMILARITY
# ============================================================================

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ============================================================================
# KNOWN FACE DATABASE
# ============================================================================

def _build_known_db():

    global _known_db

    exts = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    folder = Path(KNOWN_FACES_FOLDER)

    if not folder.exists():

        print(
            f"[FACEREC] Known faces folder not found: "
            f"{KNOWN_FACES_FOLDER}"
        )

        return

    images = [
        p for p in folder.iterdir()
        if p.suffix.lower() in exts
    ]

    if not images:

        print(
            f"[FACEREC] No images found in: "
            f"{KNOWN_FACES_FOLDER}"
        )

        return

    print(
        f"[FACEREC] Building known-face database "
        f"from {len(images)} image(s)..."
    )

    db = {}

    for img_path in images:

        img = cv2.imread(str(img_path))

        if img is None:
            continue

        h, w = img.shape[:2]

        print(
            f"[FACEREC] Reading: "
            f"{img_path.name} ({w}x{h} px)"
        )

        emb = _get_embedding(img)

        if emb is None:

            print(
                f"[FACEREC] [!] No face detected in: "
                f"{img_path.name}"
            )

            continue

        name = (
            img_path.stem
            .replace("_", " ")
            .replace("-", " ")
            .title()
        )

        db[name] = emb

        print(
            f"[FACEREC] Enrolled: {name}"
        )

    with _db_lock:
        _known_db = db

    if db:

        print(
            f"[FACEREC] Database ready — "
            f"{len(db)} person(s): "
            f"{', '.join(db)}\n"
        )

    else:

        print(
            "[FACEREC] Warning: "
            "No faces enrolled in database.\n"
        )


# ============================================================================
# LOAD EXISTING UNKNOWN FACES
# ============================================================================

def _load_existing_unknowns():

    global _unknown_db
    global _unknown_counter

    exts = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    folder = Path(UNKNOWN_FACES_FOLDER)

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    images = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in exts
    )

    if not images:

        _unknown_counter = 0

        return

    print(
        f"[FACEREC] Loading "
        f"{len(images)} existing unknown face(s) "
        f"for dedup..."
    )

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

        try:

            idx = int(
                img_path.stem.split("_")[0]
            )

            max_index = max(
                max_index,
                idx
            )

        except ValueError:

            pass

    with _unknown_lock:

        _unknown_db = db

        _unknown_counter = max_index + 1

    print(
        f"[FACEREC] Unknown-face dedup "
        f"primed with {len(db)} existing face(s).\n"
    )


# ============================================================================
# UNKNOWN FACE COMPARISON
# ============================================================================

def _find_similar_unknown(emb):

    with _unknown_lock:

        best_filename = None

        best_similarity = -1.0

        for filename, saved_emb in _unknown_db.items():

            sim = _cosine(
                emb,
                saved_emb
            )

            if sim > best_similarity:

                best_similarity = sim

                best_filename = filename

        if (
            best_filename is not None
            and
            best_similarity >= UNKNOWN_DEDUP_THRESHOLD
        ):

            return (
                best_filename,
                best_similarity
            )

        return (
            None,
            best_similarity
        )


# ============================================================================
# SAVE UNKNOWN FACE
# ============================================================================

def _save_unknown(face_crop, emb):

    global _unknown_counter

    os.makedirs(
        UNKNOWN_FACES_FOLDER,
        exist_ok=True
    )

    with _unknown_lock:

        filename = (
            f"{_unknown_counter}_unknown.jpg"
        )

        path = os.path.join(
            UNKNOWN_FACES_FOLDER,
            filename
        )

        success = cv2.imwrite(
            path,
            face_crop
        )

        if not success:

            raise RuntimeError(
                f"Failed to save intruder image: {path}"
            )

        _unknown_db[filename] = emb

        _unknown_counter += 1

    return path


# ============================================================================
# SEND INTRUSION TO WEBSITE
# ============================================================================

def _report_intrusion(image_path, similarity):

    if not INTRUSION_REPORT_URL:

        print(
            "[SECURITY] INTRUSION_REPORT_URL not configured."
        )

        return

    if not INTRUSION_PUSH_TOKEN:

        print(
            "[SECURITY] INTRUSION_PUSH_TOKEN not configured."
        )

        return

    try:

        print(
            "[SECURITY] Sending intruder image "
            "to website..."
        )

        with open(
            image_path,
            "rb"
        ) as image_file:

            files = {
                "image": (
                    os.path.basename(image_path),
                    image_file,
                    "image/jpeg"
                )
            }

            data = {
                "confidence": "85",
                "similarity": str(similarity)
            }

            headers = {
                "X-Intrusion-Token":
                INTRUSION_PUSH_TOKEN
            }

            response = requests.post(
                INTRUSION_REPORT_URL,
                files=files,
                data=data,
                headers=headers,
                timeout=INTRUSION_REPORT_TIMEOUT
            )

        if response.ok:

            print(
                "[SECURITY] Intrusion successfully "
                "reported to website."
            )

        else:

            print(
                f"[SECURITY] Website rejected "
                f"intrusion report: "
                f"{response.status_code} "
                f"{response.text}"
            )

    except Exception as e:

        print(
            f"[SECURITY] Failed to report intrusion: "
            f"{e}"
        )


def _report_intrusion_async(
    image_path,
    similarity
):

    thread = threading.Thread(
        target=_report_intrusion,
        args=(
            image_path,
            similarity
        ),
        daemon=True
    )

    thread.start()


# ============================================================================
# FACE IDENTIFICATION
# ============================================================================

def identify_face(face_crop: np.ndarray) -> dict:

    result = {

        "status": "ERROR",

        "matched_person": "UNKNOWN",

        "similarity": 0.0,

        "saved_path": None,

        "top_candidates": [],

        "error": "",
    }

    if (
        face_crop is None
        or
        face_crop.size == 0
    ):

        result["error"] = "Empty face crop"

        return result

    # ------------------------------------------------------------------------
    # Generate embedding
    # ------------------------------------------------------------------------

    emb = _get_embedding(
        face_crop
    )

    if emb is None:

        result["status"] = "NO_FACE"

        result["error"] = (
            "No face detected in crop"
        )

        return result

    # ------------------------------------------------------------------------
    # Check known people
    # ------------------------------------------------------------------------

    with _db_lock:

        known_items = list(
            _known_db.items()
        )

    if known_items:

        comparisons = [

            {
                "person_name": name,
                "similarity": round(
                    _cosine(
                        emb,
                        known_emb
                    ),
                    4
                )
            }

            for name, known_emb
            in known_items

        ]

        comparisons_sorted = sorted(
            comparisons,
            key=lambda r:
            r["similarity"],
            reverse=True
        )

        best = comparisons_sorted[0]

        result["top_candidates"] = (
            comparisons_sorted[:3]
        )

        if (
            best["similarity"]
            >= MATCH_THRESHOLD
        ):

            result["status"] = "MATCHED"

            result["matched_person"] = (
                best["person_name"]
            )

            result["similarity"] = (
                best["similarity"]
            )

            return result

    # ------------------------------------------------------------------------
    # Check previously detected unknown person
    # ------------------------------------------------------------------------

    dup_filename, dup_similarity = (
        _find_similar_unknown(emb)
    )

    if dup_filename is not None:

        result["status"] = "DUPLICATE"

        result["matched_person"] = (
            dup_filename
        )

        result["similarity"] = (
            dup_similarity
        )

        return result

    # ------------------------------------------------------------------------
    # NEW UNKNOWN PERSON
    # ------------------------------------------------------------------------

    saved_path = _save_unknown(
        face_crop,
        emb
    )

    result["status"] = "NEW_UNKNOWN"

    result["saved_path"] = saved_path

    result["similarity"] = (
        dup_similarity
        if dup_similarity > 0
        else 0.0
    )

    # ------------------------------------------------------------------------
    # REPORT TO WEBSITE
    # ------------------------------------------------------------------------

    _report_intrusion_async(
        saved_path,
        result["similarity"]
    )

    print(
        f"[SECURITY] NEW INTRUDER: "
        f"{saved_path}"
    )

    return result


# ============================================================================
# PUBLIC HELPERS
# ============================================================================

def get_known_faces():

    with _db_lock:

        return list(
            _known_db.keys()
        )


def get_unknown_face_count():

    with _unknown_lock:

        return len(
            _unknown_db
        )