import os
import shutil
from pathlib import Path

# Define base paths
BASE_DIR = Path(r"D:\downloads_v2\academic docs\tukl_internship\dataset")
SRC_DIR = BASE_DIR / "official_datasets_nwrd_frdi"
DST_DIR = BASE_DIR / "FRDIxNWRD_hybrid"

# Mapping source folders to prefixes
DATASETS = [
    {"folder_name": "NWRD", "prefix": "nwrd_"},
    {"folder_name": "NWRDF", "prefix": "frdi_"}
]

SPLITS = ["train", "val", "test"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def create_hybrid_dataset():
    counts = {split: {"images": 0, "masks": 0, "breakdown": {}} for split in SPLITS}
    
    print("🚀 Starting Hybrid Dataset Creation...\n")

    for split in SPLITS:
        print(f"================ processing [{split.upper()}] ================")
        
        # Target destination folders
        target_img_dir = DST_DIR / split / "images"
        target_mask_dir = DST_DIR / split / "masks"
        
        # Ensure output directories exist
        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_mask_dir.mkdir(parents=True, exist_ok=True)
        
        split_img_total = 0
        split_mask_total = 0

        for ds in DATASETS:
            folder_name = ds["folder_name"]
            prefix = ds["prefix"]
            
            src_img_dir = SRC_DIR / folder_name / split / "images"
            src_mask_dir = SRC_DIR / folder_name / split / "masks"

            if not src_img_dir.exists():
                print(f"⚠️  Directory missing: {src_img_dir}. Skipping...")
                continue

            # Gather image files
            images = [f for f in src_img_dir.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
            
            added_imgs = 0
            added_masks = 0

            for img_path in images:
                # 1. Copy Image with Prefix
                new_img_name = f"{prefix}{img_path.name}"
                dst_img_path = target_img_dir / new_img_name
                shutil.copy2(img_path, dst_img_path)
                added_imgs += 1

                # 2. Find and Copy Corresponding Mask
                corresponding_mask = src_mask_dir / img_path.name
                
                # Check for stem match if exact filename doesn't exist (e.g. img.jpg -> img.png)
                if not corresponding_mask.exists():
                    mask_matches = list(src_mask_dir.glob(f"{img_path.stem}.*"))
                    if mask_matches:
                        corresponding_mask = mask_matches[0]

                if corresponding_mask.exists():
                    new_mask_name = f"{prefix}{corresponding_mask.name}"
                    dst_mask_path = target_mask_dir / new_mask_name
                    shutil.copy2(corresponding_mask, dst_mask_path)
                    added_masks += 1
                else:
                    print(f"⚠️  Warning: Missing mask for {img_path.name} in {folder_name}/{split}")

            print(f"  ➜ [{folder_name}] Added {added_imgs} images | {added_masks} masks")
            
            split_img_total += added_imgs
            split_mask_total += added_masks
            counts[split]["breakdown"][folder_name] = {"images": added_imgs, "masks": added_masks}

        counts[split]["images"] = split_img_total
        counts[split]["masks"] = split_mask_total
        print(f"Subtotal for {split.upper()}: {split_img_total} images | {split_mask_total} masks\n")

    # Final Summary Output
    print("\n" + "=" * 45)
    print("      📊 HYBRID DATASET COUNT REPORT      ")
    print("=" * 45)
    for split in SPLITS:
        print(f"📁 {split.upper()} SPLIT:")
        print(f"   • Total Images : {counts[split]['images']}")
        print(f"   • Total Masks  : {counts[split]['masks']}")
        for ds_name, details in counts[split]["breakdown"].items():
            print(f"     └─ {ds_name}: {details['images']} imgs / {details['masks']} masks")
        print("-" * 45)

if __name__ == "__main__":
    create_hybrid_dataset()