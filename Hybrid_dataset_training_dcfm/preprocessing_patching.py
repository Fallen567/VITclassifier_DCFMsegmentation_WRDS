import os
import cv2
import numpy as np
from pathlib import Path

# ==========================================
# 1. PATHS & HYPERPARAMETERS
# ==========================================
HYBRID_DATASET_DIR = Path(r"D:\downloads_v2\academic docs\tukl_internship\dataset\FRDIxNWRD_hybrid")
SPLITS = ["train", "val", "test"]

PATCH_SIZE = 224
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def find_matching_mask(masks_dir, stem):
    """Locate matching mask file even if file extension differs."""
    for ext in VALID_EXTENSIONS:
        candidate = masks_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None

def patch_dataset():
    print("🚀 Starting Patch Generation for Hybrid Dataset...\n")
    
    total_summary = {}

    for split in SPLITS:
        split_dir = HYBRID_DATASET_DIR / split
        raw_images_dir = split_dir / "images"
        raw_masks_dir = split_dir / "masks"

        if not raw_images_dir.exists():
            print(f"⚠️  Directory missing: {raw_images_dir}. Skipping...")
            continue

        # Target patch output directories (Inside the same split folder)
        patch_img_dir = split_dir / "image_patches"
        patch_mask_dir = split_dir / "mask_patches"

        patch_img_dir.mkdir(parents=True, exist_ok=True)
        patch_mask_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted([f for f in raw_images_dir.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS])
        
        total_patches_generated = 0
        processed_images = 0

        print(f"================ Processing [{split.upper()}] ================")

        for img_path in image_files:
            stem = img_path.stem
            mask_path = find_matching_mask(raw_masks_dir, stem)

            if not mask_path:
                print(f"⚠️  No matching mask found for '{img_path.name}'. Skipping.")
                continue

            # Read full-size image and mask
            image = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                print(f"❌ Error reading image/mask for '{stem}'. Skipping.")
                continue

            h, w, _ = image.shape
            patch_idx = 0

            # Grid slicing without overlap (Stride = 224)
            for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
                for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
                    img_patch = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                    mask_patch = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

                    patch_filename = f"{stem}_patch_{patch_idx}.png"

                    # Save image patch and corresponding mask patch
                    cv2.imwrite(str(patch_img_dir / patch_filename), img_patch)
                    cv2.imwrite(str(patch_mask_dir / patch_filename), mask_patch)

                    patch_idx += 1
                    total_patches_generated += 1

            processed_images += 1

        print(f"  ➜ Processed {processed_images} full-size images.")
        print(f"  ➜ Generated {total_patches_generated} total 224x224 patches.\n")
        total_summary[split] = total_patches_generated

    # Final Summary Report
    print("=" * 45)
    print("      📊 PATCH GENERATION REPORT      ")
    print("=" * 45)
    for split, count in total_summary.items():
        print(f"📁 {split.upper()} SPLIT:")
        print(f"   • Total Patches Created : {count}")
        print(f"   • Folder structure      : {HYBRID_DATASET_DIR / split / 'image_patches'}")
        print(f"                           : {HYBRID_DATASET_DIR / split / 'mask_patches'}")
        print("-" * 45)

if __name__ == "__main__":
    patch_dataset()