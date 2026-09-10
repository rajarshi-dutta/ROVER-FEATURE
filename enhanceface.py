import cv2
import numpy as np
import os

# ---------- CONFIG ----------
CROPS_FOLDER = r"C:\Users\Rajarshi\OneDrive\Desktop\Robo dog\results\face_crops"
OUTPUT_FOLDER = r"C:\Users\Rajarshi\OneDrive\Desktop\Robo dog\results\fresh_face"
ENHANCEMENT_METHOD = 4  # Choose method: 1-8 (see options below)
# 1 = Basic (your current), 2 = Bilateral, 3 = Contrast, 4 = Unsharp, 
# 5 = Multi-Channel, 6 = Multi-Scale, 7 = Morphological, 8 = Aggressive
# -----------------------------


class FaceEnhancer:
    """Advanced face image enhancement methods"""
    
    @staticmethod
    def method_1_basic(face_crop):
        """Method 1: Your current approach - Denoise + CLAHE + Adaptive Gamma"""
        denoised = cv2.fastNlMeansDenoisingColored(
            face_crop, None,
            h=5, hColor=5,
            templateWindowSize=5,
            searchWindowSize=11
        )
        
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        mean_brightness = enhanced.mean()
        if mean_brightness < 15:
            gamma = 0.25
        elif mean_brightness < 40:
            gamma = 0.45
        elif mean_brightness < 80:
            gamma = 0.7
        else:
            gamma = 1.0
        
        norm = enhanced / 255.0
        gamma_corrected = np.power(norm, gamma) * 255.0
        return gamma_corrected.astype(np.uint8), mean_brightness, gamma
    
    @staticmethod
    def method_2_bilateral(face_crop):
        """Method 2: Bilateral Filter + CLAHE + Gamma
        Better denoising while preserving edges"""
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
        gamma_corrected = np.power(norm, gamma) * 255.0
        return gamma_corrected.astype(np.uint8), mean_brightness, gamma
    
    @staticmethod
    def method_3_contrast_stretch(face_crop):
        """Method 3: Contrast Stretching + CLAHE
        Stretches intensity values to full range for better contrast"""
        img_float = face_crop.astype(np.float32)
        
        min_val = np.percentile(img_float, 2)
        max_val = np.percentile(img_float, 98)
        
        stretched = np.clip(
            (img_float - min_val) / (max_val - min_val + 1e-5) * 255, 
            0, 255
        )
        stretched = stretched.astype(np.uint8)
        
        lab = cv2.cvtColor(stretched, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        
        mean_brightness = enhanced.mean()
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR), mean_brightness, 1.0
    
    @staticmethod
    def method_4_unsharp_mask(face_crop):
        """Method 4: Unsharp Masking + CLAHE
        Sharpens edges and details for better recognition"""
        blurred = cv2.GaussianBlur(face_crop, (0, 0), 2.0)
        sharpened = cv2.addWeighted(face_crop, 1.5, blurred, -0.5, 0)
        
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        mean_brightness = enhanced.mean()
        gamma = 0.35 if mean_brightness < 50 else 0.5
        
        norm = enhanced / 255.0
        gamma_corrected = np.power(norm, gamma) * 255.0
        return gamma_corrected.astype(np.uint8), mean_brightness, gamma
    
    @staticmethod
    def method_5_multichannel_clahe(face_crop):
        """Method 5: Multi-Channel CLAHE
        Apply CLAHE to each color channel separately"""
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        
        b, g, r = cv2.split(face_crop)
        b_clahe = clahe.apply(b)
        g_clahe = clahe.apply(g)
        r_clahe = clahe.apply(r)
        
        enhanced = cv2.merge((b_clahe, g_clahe, r_clahe))
        
        mean_brightness = enhanced.mean()
        gamma = 0.35 if mean_brightness < 50 else 0.5
        
        norm = enhanced / 255.0
        gamma_corrected = np.power(norm, gamma) * 255.0
        return gamma_corrected.astype(np.uint8), mean_brightness, gamma
    
    @staticmethod
    def method_6_multiscale(face_crop):
        """Method 6: Multi-Scale Enhancement
        Process at multiple scales and blend results"""
        # Full scale
        enhanced_full = FaceEnhancer.method_1_basic(face_crop)[0]
        
        # Half scale
        small = cv2.resize(face_crop, (face_crop.shape[1]//2, face_crop.shape[0]//2))
        enhanced_small, _, _ = FaceEnhancer.method_3_contrast_stretch(small)
        enhanced_half = cv2.resize(enhanced_small, (face_crop.shape[1], face_crop.shape[0]))
        
        # Blend
        result = cv2.addWeighted(enhanced_full, 0.6, enhanced_half, 0.4, 0)
        
        mean_brightness = result.mean()
        return result.astype(np.uint8), mean_brightness, 1.0
    
    @staticmethod
    def method_7_morphological(face_crop):
        """Method 7: Morphological Enhancement
        Uses erosion/dilation to enhance structures"""
        enhanced, brightness, gamma = FaceEnhancer.method_1_basic(face_crop)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
        
        return opened.astype(np.uint8), brightness, gamma
    
    @staticmethod
    def method_8_aggressive(face_crop):
        """Method 8: Aggressive Enhancement
        Best for extremely dark images - combines all techniques"""
        # Denoise
        denoised = cv2.bilateralFilter(face_crop, d=5, sigmaColor=75, sigmaSpace=75)
        
        # Contrast stretch with aggressive percentiles
        img_float = denoised.astype(np.float32)
        min_val = np.percentile(img_float, 1)
        max_val = np.percentile(img_float, 99)
        stretched = np.clip(
            (img_float - min_val) / (max_val - min_val + 1e-5) * 255, 
            0, 255
        )
        stretched = stretched.astype(np.uint8)
        
        # CLAHE with high clip limit
        lab = cv2.cvtColor(stretched, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(6, 6))
        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Aggressive gamma
        mean_brightness = enhanced.mean()
        gamma = 0.25
        
        norm = enhanced / 255.0
        gamma_corrected = np.power(norm, gamma) * 255.0
        return gamma_corrected.astype(np.uint8), mean_brightness, gamma


def enhance_with_method(face_crop, method_id):
    """Wrapper to call specific enhancement method"""
    methods = {
        1: FaceEnhancer.method_1_basic,
        2: FaceEnhancer.method_2_bilateral,
        3: FaceEnhancer.method_3_contrast_stretch,
        4: FaceEnhancer.method_4_unsharp_mask,
        5: FaceEnhancer.method_5_multichannel_clahe,
        6: FaceEnhancer.method_6_multiscale,
        7: FaceEnhancer.method_7_morphological,
        8: FaceEnhancer.method_8_aggressive,
    }
    
    if method_id not in methods:
        print(f"Invalid method {method_id}. Using method 1 (basic).")
        method_id = 1
    
    return methods[method_id](face_crop)


# Main processing
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"\n{'='*60}")
print(f"Face Enhancement - Method {ENHANCEMENT_METHOD}")
print(f"{'='*60}")

method_names = {
    1: "Basic (Denoise + CLAHE + Gamma)",
    2: "Bilateral (Better edge preservation)",
    3: "Contrast Stretch (Better contrast)",
    4: "Unsharp Mask (Sharp details)",
    5: "Multi-Channel CLAHE (Color preservation)",
    6: "Multi-Scale (Balanced quality)",
    7: "Morphological (Structural enhancement)",
    8: "Aggressive (Very dark images)",
}

print(f"Using: {method_names[ENHANCEMENT_METHOD]}")
print(f"Processing faces from: {CROPS_FOLDER}")
print(f"Saving to: {OUTPUT_FOLDER}\n")

processed = 0
total_brightness = 0

for filename in os.listdir(CROPS_FOLDER):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    crop_path = os.path.join(CROPS_FOLDER, filename)
    face_crop = cv2.imread(crop_path)

    if face_crop is None:
        print(f"✗ Could not read {filename}, skipping")
        continue

    enhanced_face, brightness, gamma_used = enhance_with_method(face_crop, ENHANCEMENT_METHOD)

    output_path = os.path.join(OUTPUT_FOLDER, filename)
    cv2.imwrite(output_path, enhanced_face)

    print(f"✓ {filename}: brightness={brightness:.1f}, gamma={gamma_used:.2f}")
    processed += 1
    total_brightness += brightness

print(f"\n{'='*60}")
print(f"Done! {processed} enhanced face(s) saved")
print(f"Average brightness: {total_brightness/max(processed, 1):.1f}")
print(f"Output folder: {OUTPUT_FOLDER}")
print(f"{'='*60}\n")