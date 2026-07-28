import os
import shutil
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 1. DYNAMIC PATH CONFIGURATION
# ==========================================
# Resolves paths automatically relative to this script location
SCRIPT_DIR = Path(__file__).parent.resolve()

# Script location: .../tukl/hammad_asim_kayani/DCFM_vit/VITclassifier_DCFMsegmentation_WRDS/Hybrid_dataset_training_dcfm
DCFM_VIT_DIR = SCRIPT_DIR.parents[1]  # Points to .../DCFM_vit
TUKL_ROOT = SCRIPT_DIR.parents[4]     # Points to .../tukl

BASE_PREP_DIR = TUKL_ROOT / "Rudaina" / "dataset prep"
OUTPUT_HYBRID_DIR = DCFM_VIT_DIR / "hybrid_dataset"

# Fallback explicit checks if directory structure varies
if not BASE_PREP_DIR.exists():
    BASE_PREP_DIR = Path("/media/tukl/aab43c5c-a42e-46d5-affb-0a14d0d1a0b8/tukl/Rudaina/dataset prep")

if not OUTPUT_HYBRID_DIR.parent.exists():
    OUTPUT_HYBRID_DIR = Path("/media/tukl/aab43c5c-a42e-46d5-affb-0a14d0d1a0b8/tukl/hammad_asim_kayani/DCFM_vit/hybrid_dataset")

DATASETS = {
    "NWRD": "nwrd",
    "NWRDF": "frdi"
}

SPLITS = ["train", "val", "test"]

print(f"📁 Source Dataset Dir : {BASE_PREP_DIR}")
print(f"📁 Output Hybrid Dir : {OUTPUT_HYBRID_DIR}\n")

if not BASE_PREP_DIR.exists():
    raise FileNotFoundError(f"❌ Source path does not exist: {BASE_PREP_DIR}")

# ==========================================
# 2. PROCESSING & FLATTENING PATCHES
# ==========================================
def build_hybrid_dataset():
    total_images_copied = 0
    total_masks_copied = 0

    for split in SPLITS:
        print(f"================ PROCESSING SPLIT [{split.upper()}] ================")
        
        out_img_dir = OUTPUT_HYBRID_DIR / split / "image_patches"
        out_mask_dir = OUTPUT_HYBRID_DIR / split / "mask_patches"
        
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_mask_dir.mkdir(parents=True, exist_ok=True)

        for ds_folder, prefix in DATASETS.items():
            ds_split_path = BASE_PREP_DIR / ds_folder / split
            img_root = ds_split_path / "images"
            mask_root = ds_split_path / "masks"

            if not img_root.exists() or not mask_root.exists():
                print(f"⚠️ Warning: Missing path {ds_split_path}. Skipping...")
                continue

            subfolders = [f for f in img_root.iterdir() if f.is_dir()]
            print(f"  • {ds_folder} ({split}): Found {len(subfolders)} patch subfolders.")

            for subfolder in tqdm(subfolders, desc=f"    Processing {ds_folder}"):
                subfolder_name = subfolder.name
                mask_subfolder = mask_root / subfolder_name

                img_patches = list(subfolder.glob("*.*"))
                for img_patch in img_patches:
                    if img_patch.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                        continue

                    patch_filename = img_patch.name
                    patch_stem = img_patch.stem

                    new_img_name = f"{prefix}_{subfolder_name}_{patch_filename}"
                    
                    target_img_path = out_img_dir / new_img_name
                    shutil.copy2(img_patch, target_img_path)
                    total_images_copied += 1

                    target_mask_path = out_mask_dir / new_img_name
                    
                    candidate_mask = mask_subfolder / patch_filename
                    if not candidate_mask.exists():
                        candidate_mask = mask_subfolder / f"{patch_stem}.png"

                    if candidate_mask.exists():
                        shutil.copy2(candidate_mask, target_mask_path)
                        total_masks_copied += 1
                    else:
                        print(f"⚠️ Mask missing for patch: {img_patch}")

        print(f"✅ Split [{split.upper()}] complete.\n")

    print(f"🎉 Hybrid Dataset created successfully!")
    print(f"   • Total Image Patches: {total_images_copied}")
    print(f"   • Total Mask Patches : {total_masks_copied}")

if __name__ == "__main__":
    build_hybrid_dataset()