import os
import sys
import cv2
import torch
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# 1. PATHS & DIRECTORY SETUP
# ==========================================
# Script parent directory: Hybrid_dataset_training_dcfm
SCRIPT_DIR = Path(__file__).parent.resolve()

# Output folder for grouped DCFM input
DCFM_OUT_DIR = SCRIPT_DIR / "rust_classified_patches_of_hybrid_dataset"

# CrossViT directory (sibling folder to Hybrid_dataset_training_dcfm)
CROSSVIT_DIR = SCRIPT_DIR.parent / "crossvit"
sys.path.append(str(CROSSVIT_DIR))
WEIGHTS_PATH = CROSSVIT_DIR / "22.pth"

# Original Hybrid Dataset path
HYBRID_DATASET_DIR = Path(r"D:\downloads_v2\academic docs\tukl_internship\dataset\FRDIxNWRD_hybrid")
SPLITS = ["train", "val", "test"]

# Parameters matching research paper specifications
RUST_THRESHOLD = 150  # GT threshold: >= 150 rust pixels = Class 1 (Rust)
GROUP_SIZE = 12       # Paper specification: N = 12 for DCFM
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformations required by CrossViT
vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 2. LOAD PRETRAINED CROSSVIT (22.pth)
# ==========================================
def load_crossvit():
    print(f"🚀 Loading CrossViT model from: {WEIGHTS_PATH}")
    print(f"🖥️ Using device: {DEVICE}")

    try:
        import models.crossvit as crossvit_models
        model = crossvit_models.crossvit_15_224(num_classes=2)
    except Exception:
        import crossvit as crossvit_models
        model = crossvit_models.crossvit_15_224(num_classes=2)

    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    print("✅ Pretrained CrossViT loaded successfully!\n")
    return model

# ==========================================
# 3. EVALUATE & GROUP PATCHES FOR DCFM
# ==========================================
def process_and_evaluate(model):
    print(f"📁 Output directory for DCFM groups: {DCFM_OUT_DIR}\n")

    for split in SPLITS:
        split_dir = HYBRID_DATASET_DIR / split
        img_patch_dir = split_dir / "image_patches"
        mask_patch_dir = split_dir / "mask_patches"

        if not img_patch_dir.exists() or not mask_patch_dir.exists():
            print(f"⚠️ Missing patch directories for split: {split}. Skipping...")
            continue

        # Target directories inside rust_classified_patches_of_hybrid_dataset
        split_out_img_dir = DCFM_OUT_DIR / split / "images"
        split_out_mask_dir = DCFM_OUT_DIR / split / "masks"

        # Clear previous run outputs for this split if present
        if (DCFM_OUT_DIR / split).exists():
            shutil.rmtree(DCFM_OUT_DIR / split)

        split_out_img_dir.mkdir(parents=True, exist_ok=True)
        split_out_mask_dir.mkdir(parents=True, exist_ok=True)

        img_files = sorted(list(img_patch_dir.glob("*.png")))
        
        y_true = []
        y_pred = []
        vit_rust_patches = []
        results = []

        print(f"================ EVALUATING ViT ON [{split.upper()}] ================")
        print(f"Total patches to process: {len(img_files)}")

        for img_path in img_files:
            filename = img_path.name
            mask_path = mask_patch_dir / filename

            if not mask_path.exists():
                continue

            # 1. Ground Truth label calculation from mask patch
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            rust_pixel_count = np.count_nonzero(mask > 128)
            gt_label = 1 if rust_pixel_count >= RUST_THRESHOLD else 0
            y_true.append(gt_label)

            # 2. ViT Inference on RGB patch
            pil_img = Image.open(img_path).convert("RGB")
            tensor_img = vit_transform(pil_img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = model(tensor_img)
                pred_label = torch.argmax(outputs, dim=1).item()

            y_pred.append(pred_label)

            # Collect ViT-classified RUST (Class 1) patches for DCFM
            if pred_label == 1:
                vit_rust_patches.append((filename, img_path, mask_path))

            results.append({
                "patch_name": filename,
                "rust_pixels": rust_pixel_count,
                "ground_truth": gt_label,
                "vit_prediction": pred_label
            })

        # 3. Print evaluation metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        print(f"\n📊 [{split.upper()}] ViT EVALUATION METRICS:")
        print(f"   • Accuracy  : {acc * 100:.2f}%")
        print(f"   • Precision : {prec * 100:.2f}%")
        print(f"   • Recall    : {rec * 100:.2f}%")
        print(f"   • F1-Score  : {f1 * 100:.2f}%")
        print(f"   • Confusion Matrix [TN, FP / FN, TP]:\n{cm}\n")

        # Save CSV log inside hybrid dataset split folder
        df = pd.DataFrame(results)
        df.to_csv(split_dir / "vit_evaluation_results.csv", index=False)

        # 4. Group ViT-classified Rust patches into DCFM Co-Saliency Groups (N = 12)
        print(f"📦 Grouping {len(vit_rust_patches)} ViT-classified Rust patches into N={GROUP_SIZE} folders...")
        
        groups = {}
        for fname, src_img, src_mask in vit_rust_patches:
            parent_id = fname.rsplit('_patch_', 1)[0]
            if parent_id not in groups:
                groups[parent_id] = []
            groups[parent_id].append((fname, src_img, src_mask))

        group_count = 0
        for parent_id, items in groups.items():
            for i in range(0, len(items), GROUP_SIZE):
                sub_group = items[i:i + GROUP_SIZE]
                group_folder_name = f"{parent_id}_part{i // GROUP_SIZE + 1}"

                grp_img_path = split_out_img_dir / group_folder_name
                grp_mask_path = split_out_mask_dir / group_folder_name
                grp_img_path.mkdir(parents=True, exist_ok=True)
                grp_mask_path.mkdir(parents=True, exist_ok=True)

                for fname, src_img, src_mask in sub_group:
                    shutil.copy(src_img, grp_img_path / fname)
                    shutil.copy(src_mask, grp_mask_path / fname)

                group_count += 1

        print(f"✅ Saved {group_count} co-saliency subfolders to: {DCFM_OUT_DIR / split}\n")

if __name__ == "__main__":
    vit_model = load_crossvit()
    process_and_evaluate(vit_model)
    print("🎉 Pipeline Complete! DCFM input folders are structured and ready.")