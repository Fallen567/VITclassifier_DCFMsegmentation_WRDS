import os
import glob
import cv2
import shutil
import numpy as np

# ==========================================
# 1. PATHS & HYPERPARAMETERS
# ==========================================
DATA_DIR = os.path.join("data", "NWRD_test")

RAW_IMAGES_DIR = os.path.join(DATA_DIR, "images")
RAW_MASKS_DIR = os.path.join(DATA_DIR, "masks")

# Target output directories
RUST_DIR = os.path.join(DATA_DIR, "rust")
NON_RUST_DIR = os.path.join(DATA_DIR, "non_rust")

PATCH_SIZE = 224
RUST_THRESHOLD = 150  # Minimum positive pixels to count as rust
GROUP_SIZE = 12        # Co-saliency group size N = 12

os.makedirs(RUST_DIR, exist_ok=True)
os.makedirs(NON_RUST_DIR, exist_ok=True)

# Helper function to find mask regardless of file extension (.png vs .jpg)
def find_mask_file(masks_dir, stem):
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG']
    for ext in valid_extensions:
        candidate = os.path.join(masks_dir, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None

# ==========================================
# 2. PATCH CROPPING & THRESHOLDING
# ==========================================
def create_and_sort_patches():
    print("--- Step 1: Slicing 224x224 patches and thresholding ---")
    
    if not os.path.exists(RAW_IMAGES_DIR):
        print(f"Error: Could not find raw images folder at '{RAW_IMAGES_DIR}'")
        return

    image_files = sorted([f for f in os.listdir(RAW_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    total_rust = 0
    total_non_rust = 0

    for img_name in image_files:
        img_path = os.path.join(RAW_IMAGES_DIR, img_name)
        stem = os.path.splitext(img_name)[0]  # Extracts base name without .jpg/.png
        
        # Look for matching mask file with any extension
        mask_path = find_mask_file(RAW_MASKS_DIR, stem)

        if not mask_path:
            print(f"Warning: No matching mask found for '{img_name}' in {RAW_MASKS_DIR}. Skipping.")
            continue

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Error reading image or mask for '{stem}'. Skipping.")
            continue

        h, w, _ = image.shape

        patch_idx = 0
        for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
            for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
                img_patch = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                mask_patch = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

                positive_pixels = np.count_nonzero(mask_patch > 128)
                patch_filename = f"{stem}_patch_{patch_idx}.png"

                if positive_pixels >= RUST_THRESHOLD:
                    cv2.imwrite(os.path.join(RUST_DIR, patch_filename), img_patch)
                    total_rust += 1
                else:
                    cv2.imwrite(os.path.join(NON_RUST_DIR, patch_filename), img_patch)
                    total_non_rust += 1

                patch_idx += 1

    print(f"\nPatches Generated Successfully:")
    print(f"  - Rust Patches (Class 1): {total_rust}")
    print(f"  - Non-Rust Patches (Class 0): {total_non_rust}")

# ==========================================
# 3. DCFM GROUPING (N = 12)
# ==========================================
def organize_cosaliency_groups():
    print("\n--- Step 2: Creating DCFM Co-Saliency Groups (N = 12) ---")
    cosaliency_dir = os.path.join(DATA_DIR, "cosaliency", "images")
    os.makedirs(cosaliency_dir, exist_ok=True)

    rust_patches = [f for f in os.listdir(RUST_DIR) if f.endswith('.png')]
    
    groups = {}
    for filename in rust_patches:
        parent_id = filename.split('_')[0]
        if parent_id not in groups:
            groups[parent_id] = []
        groups[parent_id].append(filename)

    for parent_id, filenames in groups.items():
        for i in range(0, len(filenames), GROUP_SIZE):
            sub_group = filenames[i:i + GROUP_SIZE]
            group_folder_name = f"{parent_id}_part{i // GROUP_SIZE + 1}"
            group_folder_path = os.path.join(cosaliency_dir, group_folder_name)
            os.makedirs(group_folder_path, exist_ok=True)

            for fname in sub_group:
                src = os.path.join(RUST_DIR, fname)
                dst = os.path.join(group_folder_path, fname)
                shutil.copy(src, dst)

    print(f"Co-saliency groups ready in '{cosaliency_dir}'")

if __name__ == '__main__':
    create_and_sort_patches()
    organize_cosaliency_groups()
    print("\nPreprocessing complete!")