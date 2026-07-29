import os
import glob
import random
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
BASE_DIR = r"D:\downloads_v2\academic docs\tukl_internship\dataset\FRDIxNWRD_hybrid\train"

IMG_PATCHES_DIR = os.path.join(BASE_DIR, "image_patches")
MASK_PATCHES_DIR = os.path.join(BASE_DIR, "mask_patches")

OUT_IMG_DIR = os.path.join(BASE_DIR, "augmented_image_patches")
OUT_MASK_DIR = os.path.join(BASE_DIR, "augmented_mask_patches")

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

RUST_PIXEL_THRESHOLD = 150  # Minimum rust pixels in mask to classify as Rust

# Set random seed for reproducible undersampling
random.seed(42)

# ==========================================
# 2. TRANSFORMATIONS FUNCTION
# ==========================================
def get_transformations(img, mask):
    """
    Applies the 6 transformations from Fig 2 (a-g) to both image and mask in lockstep.
    Returns a dictionary of {suffix: (transformed_img, transformed_mask)}.
    """
    return {
        "orig": (img, mask),
        "hflip": (TF.hflip(img), TF.hflip(mask)),
        "vflip": (TF.vflip(img), TF.vflip(mask)),
        "rot90": (TF.rotate(img, 90), TF.rotate(mask, 90)),
        "rot45": (TF.rotate(img, 45), TF.rotate(mask, 45)),
        "rot180": (TF.rotate(img, 180), TF.rotate(mask, 180)),
        "rot135": (TF.rotate(img, 135), TF.rotate(mask, 135)),
    }

# ==========================================
# 3. CLASSIFY RAW PATCHES
# ==========================================
print(f"🔍 Scanning image & mask patches from: {BASE_DIR}")

valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
img_files = [
    f for f in os.listdir(IMG_PATCHES_DIR)
    if f.lower().endswith(valid_exts)
]

rust_pairs = []
non_rust_pairs = []

for img_name in tqdm(img_files, desc="Classifying Patches"):
    img_path = os.path.join(IMG_PATCHES_DIR, img_name)
    
    # Locate corresponding mask patch
    stem = os.path.splitext(img_name)[0]
    mask_path = None
    for ext in valid_exts:
        candidate = os.path.join(MASK_PATCHES_DIR, stem + ext)
        if os.path.exists(candidate):
            mask_path = candidate
            break

    if mask_path is None:
        print(f"⚠️ Warning: Missing mask for image patch {img_name}, skipping.")
        continue

    # Load mask to check rust pixel count
    mask_pil = Image.open(mask_path).convert('L')
    mask_arr = np.array(mask_pil)
    
    rust_pixels = np.sum(mask_arr > 128)  # Count non-zero mask pixels

    if rust_pixels >= RUST_PIXEL_THRESHOLD:
        rust_pairs.append((img_path, mask_path))
    else:
        non_rust_pairs.append((img_path, mask_path))

print(f"\n📊 Initial Classification Results:")
print(f"   • Raw Rust Patches Identified: {len(rust_pairs)}")
print(f"   • Raw Non-Rust Patches Identified: {len(non_rust_pairs)}")

# ==========================================
# 4. AUGMENT RUST PATCHES (7x Expansion)
# ==========================================
print("\n⚙️ Augmenting Rust Patches (1 Original + 6 Transformations)...")

augmented_rust_count = 0

for img_path, mask_path in tqdm(rust_pairs, desc="Augmenting Rust"):
    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    
    base_stem = os.path.splitext(os.path.basename(img_path))[0]
    ext = os.path.splitext(img_path)[1]

    transforms_dict = get_transformations(img, mask)

    for suffix, (trans_img, trans_mask) in transforms_dict.items():
        out_img_name = f"{base_stem}_{suffix}{ext}"
        out_mask_name = f"{base_stem}_{suffix}{ext}"

        trans_img.save(os.path.join(OUT_IMG_DIR, out_img_name))
        trans_mask.save(os.path.join(OUT_MASK_DIR, out_mask_name))
        
        augmented_rust_count += 1

# ==========================================
# 5. UNDERSAMPLE NON-RUST (1:1 Ratio)
# ==========================================
target_nonrust_count = min(augmented_rust_count, len(non_rust_pairs))
print(f"\n⚖️ Balancing Dataset: Undersampling Non-Rust patches to {target_nonrust_count}...")

selected_non_rust = random.sample(non_rust_pairs, target_nonrust_count)

non_rust_count = 0
for img_path, mask_path in tqdm(selected_non_rust, desc="Saving Non-Rust"):
    img_name = os.path.basename(img_path)
    mask_name = os.path.basename(mask_path)

    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')

    img.save(os.path.join(OUT_IMG_DIR, img_name))
    mask.save(os.path.join(OUT_MASK_DIR, mask_name))
    
    non_rust_count += 1

# ==========================================
# 6. FINAL SUMMARY STATS
# ==========================================
total_patches = len(os.listdir(OUT_IMG_DIR))

print("\n" + "="*50)
print("🎉 AUGMENTATION AND DATASET BALANCING COMPLETE!")
print("="*50)
print(f"📁 Output Saved To:")
print(f"   • Images: {OUT_IMG_DIR}")
print(f"   • Masks:  {OUT_MASK_DIR}\n")
print(f"📈 Final Patch Breakdown in 'augmented_image_patches':")
print(f"   • Total Rust Patches:     {augmented_rust_count}")
print(f"   • Total Non-Rust Patches: {non_rust_count}")
print(f"   • Total Combined Patches: {total_patches}")
print("="*50)