"""
Complete Face Detection Pipeline
Detection → Cropping → Enhancement → Upscaling
All in one script — LOSSLESS quality throughout
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "blacknwhite.pt"
SOURCE_FOLDER = r"C:\Users\Rajarshi\OneDrive\Desktop\Robo dog\test image foldere"

# Output folders
CROPS_FOLDER    = r"C:\Users\Rajarshi\OneDrive\Desktop\Robo dog\results\face_crops"
ENHANCED_FOLDER = r"C:\Users\Rajarshi\OneDrive\Desktop\Robo dog\results\face_crops_enhanced"
ENLARGED_FOLDER = r"C:\Users\Rajarshi\OneDrive\Desktop\Robo dog\results\face_crops_enlarged"

# Processing options
ENABLE_ENHANCEMENT = True   # Enhance dark faces
ENHANCEMENT_METHOD = 4      # 2 = Bilateral (recommended), 5 = Aggressive
ENABLE_UPSCALING   = True   # Enlarge small crops
UPSCALE_FACTOR     = 3      # 3x larger
UPSCALE_METHOD     = 8      # 8 = Adaptive (auto-select best)

# ── Quality settings ──────────────────────────────────────────────────────────
# PNG  = always lossless (larger file, best for face recognition input)
# JPEG = lossy (smaller file, fine for human viewing only)
SAVE_FORMAT = "png"  # "png" or "jpeg"
JPEG_QUALITY = 100   # only used when SAVE_FORMAT = "jpeg"  (100 = max, still lossy)

CROP_PADDING = 20    # extra pixels around each detected bounding box


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _ext():
    """Return the file extension to use."""
    return ".png" if SAVE_FORMAT == "png" else ".jpg"


def _save(path: str, image: np.ndarray) -> None:
    """Save image in the configured format without quality loss."""
    if SAVE_FORMAT == "png":
        cv2.imwrite(path, image)                                        # lossless
    else:
        cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def _crop_lossless(image: np.ndarray, x1: int, y1: int,
                   x2: int, y2: int, padding: int = 0) -> np.ndarray:
    """
    Pure numpy slice — no interpolation, no quality loss.
    Adds optional padding clamped to image boundaries.
    """
    h, w = image.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return image[y1:y2, x1:x2]   # ← lossless pixel slice


# ============================================================================
# ENHANCEMENT FUNCTIONS
# ============================================================================

class FaceEnhancer:
    """Enhancement methods for dark face crops"""

    @staticmethod
    def enhance_bilateral(face_crop: np.ndarray) -> np.ndarray:
        """Bilateral filter + CLAHE + Gamma (Recommended)"""
        denoised = cv2.bilateralFilter(face_crop, d=5, sigmaColor=75, sigmaSpace=75)

        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        mean_brightness = enhanced.mean()
        gamma = 0.35 if mean_brightness < 50 else 0.5

        norm = enhanced / 255.0
        return (np.power(norm, gamma) * 255.0).astype(np.uint8)

    @staticmethod
    def enhance_aggressive(face_crop: np.ndarray) -> np.ndarray:
        """Aggressive enhancement for very dark faces"""
        denoised = cv2.bilateralFilter(face_crop, d=5, sigmaColor=75, sigmaSpace=75)

        img_float = denoised.astype(np.float32)
        min_val   = np.percentile(img_float, 1)
        max_val   = np.percentile(img_float, 99)
        stretched = np.clip(
            (img_float - min_val) / (max_val - min_val + 1e-5) * 255,
            0, 255
        ).astype(np.uint8)

        lab = cv2.cvtColor(stretched, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(6, 6))
        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        norm = enhanced / 255.0
        return (np.power(norm, 0.25) * 255.0).astype(np.uint8)

    @staticmethod
    def enhance_face(face_crop: np.ndarray, method: int = 2) -> np.ndarray:
        if method == 2:
            return FaceEnhancer.enhance_bilateral(face_crop)
        else:
            return FaceEnhancer.enhance_aggressive(face_crop)


# ============================================================================
# UPSCALING FUNCTIONS
# ============================================================================

class FaceUpscaler:
    """Upscaling methods for small face crops"""

    @staticmethod
    def upscale_lanczos(image: np.ndarray, scale_factor: int = 3) -> np.ndarray:
        h, w = image.shape[:2]
        return cv2.resize(image, (w * scale_factor, h * scale_factor),
                          interpolation=cv2.INTER_LANCZOS4)

    @staticmethod
    def upscale_sharpen(image: np.ndarray, scale_factor: int = 3) -> np.ndarray:
        upscaled = FaceUpscaler.upscale_lanczos(image, scale_factor)
        blurred  = cv2.GaussianBlur(upscaled, (0, 0), 1.0)
        sharpened = cv2.addWeighted(upscaled, 1.5, blurred, -0.5, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def upscale_multiscale(image: np.ndarray, scale_factor: int = 3) -> np.ndarray:
        result = image.copy()
        steps  = int(np.log(scale_factor) / np.log(1.5))
        for _ in range(steps):
            h, w   = result.shape[:2]
            result = cv2.resize(result, (int(w * 1.5), int(h * 1.5)),
                                interpolation=cv2.INTER_CUBIC)
        h, w = image.shape[:2]
        return cv2.resize(result, (w * scale_factor, h * scale_factor),
                          interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def upscale_adaptive(image: np.ndarray, scale_factor: int = 3) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < 100:
            return FaceUpscaler.upscale_sharpen(image, scale_factor)
        return FaceUpscaler.upscale_multiscale(image, scale_factor)

    @staticmethod
    def upscale_face(image: np.ndarray, scale_factor: int = 3,
                     method: int = 8) -> np.ndarray:
        if method == 8:
            return FaceUpscaler.upscale_adaptive(image, scale_factor)
        elif method == 5:
            return FaceUpscaler.upscale_sharpen(image, scale_factor)
        return FaceUpscaler.upscale_lanczos(image, scale_factor)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_detection_pipeline():
    """Complete face detection, cropping, enhancement, upscaling pipeline"""

    print(f"\n{'='*80}")
    print("FACE DETECTION PIPELINE")
    print(f"{'='*80}")
    print(f"  Model             : {MODEL_PATH}")
    print(f"  Source folder     : {SOURCE_FOLDER}")
    print(f"  Save format       : {SAVE_FORMAT.upper()}"
          + (f" (quality {JPEG_QUALITY})" if SAVE_FORMAT == "jpeg" else " (lossless)"))
    print(f"  Crop padding      : {CROP_PADDING} px")
    print(f"  Enhancement       : {'ON (method ' + str(ENHANCEMENT_METHOD) + ')' if ENABLE_ENHANCEMENT else 'OFF'}")
    print(f"  Upscaling         : {'ON (' + str(UPSCALE_FACTOR) + 'x, method ' + str(UPSCALE_METHOD) + ')' if ENABLE_UPSCALING else 'OFF'}")
    print(f"{'='*80}\n")

    # Create output folders
    os.makedirs(CROPS_FOLDER, exist_ok=True)
    if ENABLE_ENHANCEMENT:
        os.makedirs(ENHANCED_FOLDER, exist_ok=True)
    if ENABLE_UPSCALING:
        os.makedirs(ENLARGED_FOLDER, exist_ok=True)

    # Load model
    print("Loading YOLO model…")
    model = YOLO(MODEL_PATH)
    print("✓ Model loaded\n")

    total_images = 0
    total_faces  = 0
    ext          = _ext()

    image_files = [f for f in os.listdir(SOURCE_FOLDER)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]

    for idx, filename in enumerate(image_files, 1):
        image_path = os.path.join(SOURCE_FOLDER, filename)

        # ── Read source image ONCE at full resolution ─────────────────────────
        image = cv2.imread(image_path)
        if image is None:
            print(f"{idx}. {filename}  ✗ Cannot read — skipping\n")
            continue

        h_src, w_src = image.shape[:2]
        print(f"{idx}. {filename}  ({w_src}×{h_src} px)")

        # ── Detect ────────────────────────────────────────────────────────────
        results = model.predict(source=image, verbose=False)
        boxes   = results[0].boxes.xyxy.cpu().numpy()
        base    = os.path.splitext(filename)[0]

        print(f"   Detected: {len(boxes)} face(s)")
        if len(boxes) == 0:
            print()
            continue

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)

            # ── Step 1: Lossless crop with padding ────────────────────────────
            raw_crop = _crop_lossless(image, x1, y1, x2, y2, padding=CROP_PADDING)
            if raw_crop.size == 0:
                continue

            name      = f"{base}_face_{i}{ext}"
            crop_path = os.path.join(CROPS_FOLDER, name)
            _save(crop_path, raw_crop)   # save original crop

            # ── Step 2: Enhancement (operates on raw numpy array, no re-read) ─
            if ENABLE_ENHANCEMENT:
                enhanced = FaceEnhancer.enhance_face(raw_crop, ENHANCEMENT_METHOD)
                _save(os.path.join(ENHANCED_FOLDER, name), enhanced)
                to_upscale = enhanced          # upscale the enhanced version
            else:
                to_upscale = raw_crop

            # ── Step 3: Upscaling (operates on numpy array, no re-read) ───────
            if ENABLE_UPSCALING:
                enlarged = FaceUpscaler.upscale_face(to_upscale, UPSCALE_FACTOR,
                                                     UPSCALE_METHOD)
                _save(os.path.join(ENLARGED_FOLDER, name), enlarged)
                final_h, final_w = enlarged.shape[:2]
            else:
                final_h, final_w = to_upscale.shape[:2]

            crop_h, crop_w = raw_crop.shape[:2]
            parts = [f"   Face {i}: {crop_w}×{crop_h}"]
            if ENABLE_ENHANCEMENT:
                parts.append("Enhanced")
            if ENABLE_UPSCALING:
                parts.append(f"{UPSCALE_FACTOR}x → {final_w}×{final_h}")
            print("  →  ".join(parts))

            total_faces += 1

        total_images += 1
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"  Images processed : {total_images}")
    print(f"  Faces detected   : {total_faces}")
    print(f"  Format           : {SAVE_FORMAT.upper()}"
          + (" (lossless)" if SAVE_FORMAT == "png" else f" (quality {JPEG_QUALITY})"))
    print(f"\n  Output locations:")
    print(f"    Crops    : {CROPS_FOLDER}")
    if ENABLE_ENHANCEMENT:
        print(f"    Enhanced : {ENHANCED_FOLDER}")
    if ENABLE_UPSCALING:
        print(f"    Enlarged : {ENLARGED_FOLDER}")
    print(f"{'='*80}\n")


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    run_detection_pipeline()