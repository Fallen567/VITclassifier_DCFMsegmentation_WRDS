import os
import sys
import cv2
import torch
import shutil
import types
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# 0. HUGGINGFACE UNPICKLING PATCH (From test.py)
# ==========================================
import torch.nn as nn
import transformers.models.vit.modeling_vit as vit_module

class DummyModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

class DynamicVitModule(types.ModuleType):
    def __getattr__(self, name):
        return DummyModule

vit_module.__class__ = DynamicVitModule

# ==========================================
# 1. PATHS & DIRECTORY SETUP
# ==========================================
SCRIPT_DIR = Path(__file__).parent.resolve()
PARENT_DIR = SCRIPT_DIR.parent
CROSSVIT_DIR = PARENT_DIR / "crossvit"

# System paths
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(CROSSVIT_DIR))

# Path for DCFM input output inside Hybrid_dataset_training_dcfm
DCFM_OUT_DIR = SCRIPT_DIR / "rust_classified_patches_of_hybrid_dataset"
WEIGHTS_PATH = CROSSVIT_DIR / "22.pth"

# Original Hybrid Dataset path
HYBRID_DATASET_DIR = Path(r"D:\downloads_v2\academic docs\tukl_internship\dataset\FRDIxNWRD_hybrid")
SPLITS = ["train", "val", "test"]

RUST_THRESHOLD = 150  # GT threshold: >= 150 rust pixels = Class 1 (Rust)
GROUP_SIZE = 12       # N = 12 patches per group for DCFM
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformations for 224x224 ViT input
vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 2. LOAD PRETRAINED ViT MODEL (22.pth)
# ==========================================
def load_crossvit():
    print(f"🚀 Loading ViT model from: {WEIGHTS_PATH}")
    print(f"🖥️ Using device: {DEVICE}")

    # Load the model directly using torch.load
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        from transformers import ViTForImageClassification
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=2)
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        model.load_state_dict(state_dict)
    else:
        model = checkpoint

    model.to(DEVICE)
    model.eval()
    print("✅ Pretrained ViT model loaded successfully on GTX 1080!\n")
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

        split_out_img_dir = DCFM_OUT_DIR / split / "images"
        split_out_mask_dir = DCFM_OUT_DIR / split / "masks"

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
                # Handle Hugging Face SequenceClassifierOutput vs PyTorch Tensor
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                pred_label = torch.argmax(logits, dim=1).item()

            y_pred.append(pred_label)

            # Collect ViT-classified RUST patches (Class 1) for DCFM
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

        # Save CSV log inside split folder
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