"""
Merges WIDER FACE (normal light) and DARK FACE (low light) datasets into one combined dataset.
This creates a single training dataset that includes both conditions.

Expected input structure:
    WIDER_FACE_DATASET/
        train/
            images/
            labels/
        val/
            images/
            labels/
    
    DARK_FACE_DATASET/
        train/
            images/
            labels/
        val/
            images/
            labels/

Output structure:
    COMBINED_DATASET/
        train/
            images/  (all WIDER FACE + all DARK FACE train images)
            labels/
        val/
            images/  (all WIDER FACE + all DARK FACE val images)
            labels/
        data.yaml
"""

import os
import shutil

# ---------- CONFIG: UPDATE THESE PATHS ----------
WIDER_FACE_PATH = r"C:\Users\Rajarshi\Downloads\archive(1)\data"  # EDIT: your WIDER FACE dataset path
DARK_FACE_PATH = r"C:\Users\Rajarshi\OneDrive\Desktop\archive\yolo_dataset"  # DARK FACE dataset
OUTPUT_PATH = r"C:\Users\Rajarshi\OneDrive\Desktop\combined_face_dataset"  # where to save merged dataset
# -----------------------------------------------


def merge_datasets(wider_path, dark_path, output_path):
    """Merge two face datasets into one combined dataset."""
    
    # Create output directory structure
    os.makedirs(f"{output_path}/train/images", exist_ok=True)
    os.makedirs(f"{output_path}/train/labels", exist_ok=True)
    os.makedirs(f"{output_path}/val/images", exist_ok=True)
    os.makedirs(f"{output_path}/val/labels", exist_ok=True)
    
    print("Merging datasets...")
    train_count = 0
    val_count = 0
    
    # Copy WIDER FACE train images and labels
    print("\nCopying WIDER FACE train data...")
    wider_train_img = os.path.join(wider_path, "train", "images")
    wider_train_lbl = os.path.join(wider_path, "train", "labels")
    
    if os.path.exists(wider_train_img):
        for filename in os.listdir(wider_train_img):
            src = os.path.join(wider_train_img, filename)
            dst = os.path.join(output_path, "train", "images", filename)
            shutil.copy(src, dst)
            train_count += 1
    
    if os.path.exists(wider_train_lbl):
        for filename in os.listdir(wider_train_lbl):
            src = os.path.join(wider_train_lbl, filename)
            dst = os.path.join(output_path, "train", "labels", filename)
            shutil.copy(src, dst)
    
    print(f"  Copied {train_count} WIDER FACE train images")
    
    # Copy WIDER FACE val images and labels
    print("Copying WIDER FACE val data...")
    wider_val_img = os.path.join(wider_path, "val", "images")
    wider_val_lbl = os.path.join(wider_path, "val", "labels")
    
    wider_val_count = 0
    if os.path.exists(wider_val_img):
        for filename in os.listdir(wider_val_img):
            src = os.path.join(wider_val_img, filename)
            dst = os.path.join(output_path, "val", "images", filename)
            shutil.copy(src, dst)
            wider_val_count += 1
    
    if os.path.exists(wider_val_lbl):
        for filename in os.listdir(wider_val_lbl):
            src = os.path.join(wider_val_lbl, filename)
            dst = os.path.join(output_path, "val", "labels", filename)
            shutil.copy(src, dst)
    
    print(f"  Copied {wider_val_count} WIDER FACE val images")
    val_count += wider_val_count
    
    # Copy DARK FACE train images and labels
    print("Copying DARK FACE train data...")
    dark_train_img = os.path.join(dark_path, "train", "images")
    dark_train_lbl = os.path.join(dark_path, "train", "labels")
    
    dark_train_count = 0
    if os.path.exists(dark_train_img):
        for filename in os.listdir(dark_train_img):
            src = os.path.join(dark_train_img, filename)
            dst = os.path.join(output_path, "train", "images", filename)
            shutil.copy(src, dst)
            dark_train_count += 1
    
    if os.path.exists(dark_train_lbl):
        for filename in os.listdir(dark_train_lbl):
            src = os.path.join(dark_train_lbl, filename)
            dst = os.path.join(output_path, "train", "labels", filename)
            shutil.copy(src, dst)
    
    print(f"  Copied {dark_train_count} DARK FACE train images")
    train_count += dark_train_count
    
    # Copy DARK FACE val images and labels
    print("Copying DARK FACE val data...")
    dark_val_img = os.path.join(dark_path, "val", "images")
    dark_val_lbl = os.path.join(dark_path, "val", "labels")
    
    dark_val_count = 0
    if os.path.exists(dark_val_img):
        for filename in os.listdir(dark_val_img):
            src = os.path.join(dark_val_img, filename)
            dst = os.path.join(output_path, "val", "images", filename)
            shutil.copy(src, dst)
            dark_val_count += 1
    
    if os.path.exists(dark_val_lbl):
        for filename in os.listdir(dark_val_lbl):
            src = os.path.join(dark_val_lbl, filename)
            dst = os.path.join(output_path, "val", "labels", filename)
            shutil.copy(src, dst)
    
    print(f"  Copied {dark_val_count} DARK FACE val images")
    val_count += dark_val_count
    
    # Write combined data.yaml
    yaml_content = f"""train: {os.path.join(output_path, 'train', 'images')}
val: {os.path.join(output_path, 'val', 'images')}
nc: 1
names: ['face']
"""
    yaml_path = os.path.join(output_path, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    
    print("\n" + "="*60)
    print("MERGE COMPLETE!")
    print("="*60)
    print(f"Total train images: {train_count}")
    print(f"Total val images: {val_count}")
    print(f"Combined dataset saved to: {output_path}")
    print(f"data.yaml location: {yaml_path}")
    print("\nNext step: Train your model with this combined dataset:")
    print(f"  model.train(data='{yaml_path}', epochs=50, imgsz=640, batch=16)")


if __name__ == "__main__":
    # Verify paths exist before starting
    if not os.path.exists(WIDER_FACE_PATH):
        print(f"ERROR: WIDER FACE path not found: {WIDER_FACE_PATH}")
        print("Please update WIDER_FACE_PATH at the top of this script.")
        exit(1)
    
    if not os.path.exists(DARK_FACE_PATH):
        print(f"ERROR: DARK FACE path not found: {DARK_FACE_PATH}")
        print("Please update DARK_FACE_PATH at the top of this script.")
        exit(1)
    
    merge_datasets(WIDER_FACE_PATH, DARK_FACE_PATH, OUTPUT_PATH)