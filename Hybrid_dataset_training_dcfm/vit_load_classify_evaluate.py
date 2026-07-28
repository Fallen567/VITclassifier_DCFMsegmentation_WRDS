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
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# 0. HUGGINGFACE UNPICKLING COMPATIBILITY PATCH
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
REPO_ROOT = SCRIPT_DIR.parent       # .../VITclassifier_DCFMsegmentation_WRDS
DCFM_VIT_DIR = REPO_ROOT.parent     # .../DCFM_vit

CROSSVIT_DIR = REPO_ROOT / "crossvit"
WEIGHTS_PATH = CROSSVIT_DIR / "22.pth"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CROSSVIT_DIR))

# Input Hybrid Dataset path
HYBRID_DATASET_DIR = DCFM_VIT_DIR / "hybrid_dataset"

# Output directory on Lab PC
DCFM_OUT_DIR = DCFM_VIT_DIR / "VIT_classified_patches_augmented_hybrid_dataset"

if not HYBRID_DATASET_DIR.exists():
    HYBRID_DATASET_DIR = Path("/media/tukl/aab43c5c-a42e-46d5-affb-0a14d0d1a0b8/tukl/hammad_asim_kayani/DCFM_vit/hybrid_dataset")

if not DCFM_OUT_DIR.parent.exists():
    DCFM_OUT_DIR = Path("/media/tukl/aab43c5c-a42e-46d5-affb-0a14d0d1a0b8/tukl/hammad_asim_kayani/DCFM_vit/VIT_classified_patches_augmented_hybrid_dataset")

SPLITS = ["train", "val", "test"]

RUST_THRESHOLD = 150  # Ground Truth threshold: >= 150 rust pixels = Class 1 (Rust)
GROUP_SIZE = 12       # N = 12 patches per group for DCFM
BATCH_SIZE = 128      # Fast GPU batch inference
NUM_WORKERS = 4       # Multi-threaded image loading
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 2. DATASET CLASS FOR BATCHED INFERENCE
# ==========================================
class PatchInferenceDataset(Dataset):
    def __init__(self, img_files, mask_patch_dir, transform, rust_threshold):
        self.img_files = img_files
        self.mask_patch_dir = mask_patch_dir
        self.transform = transform
        self.rust_threshold = rust_threshold

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        filename = img_path.name
        mask_path = self.mask_patch_dir / filename

        if not mask_path.exists():
            mask_path = self.mask_patch_dir / f"{img_path.stem}.png"

        rust_pixel_count = 0
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                rust_pixel_count = int(np.count_nonzero(mask > 128))

        gt_label = 1 if rust_pixel_count >= self.rust_threshold else 0

        pil_img = Image.open(img_path).convert("RGB")
        tensor_img = self.transform(pil_img)

        return tensor_img, gt_label, rust_pixel_count, str(img_path), str(mask_path), filename

# ==========================================
# 3. LOAD WEIGHTS INTO FRESH MODERN ViT MODEL
# ==========================================
def load_crossvit():
    from transformers import ViTForImageClassification

    print(f"🚀 Loading ViT checkpoint from: {WEIGHTS_PATH}")
    print(f"🖥️ Using device: {DEVICE}")

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"❌ Weights file not found at: {WEIGHTS_PATH}")

    # 1. Unpickle legacy checkpoint
    raw_checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)

    # 2. Extract pure state_dict tensor dictionary
    if hasattr(raw_checkpoint, "state_dict"):
        state_dict = raw_checkpoint.state_dict()
    elif isinstance(raw_checkpoint, dict):
        state_dict = raw_checkpoint.get("state_dict", raw_checkpoint.get("model", raw_checkpoint))
    else:
        state_dict = raw_checkpoint

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # 3. Instantiate a fresh model under current transformers library
    print("📦 Instantiating clean modern ViT model architecture ('google/vit-base-patch16-224')...")
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    # 4. Load trained weights into fresh architecture
    load_res = model.load_state_dict(state_dict, strict=False)
    print(f"⚙️ Weights loaded successfully into fresh model. Missing keys: {len(load_res.missing_keys)}, Unexpected keys: {len(load_res.unexpected_keys)}")

    del raw_checkpoint
    del state_dict

    model.to(DEVICE)
    model.eval()
    print("✅ Pretrained ViT model ready for inference!\n")
    return model

# ==========================================
# 4. EVALUATE & GROUP PATCHES FOR DCFM
# ==========================================
def process_and_evaluate(model):
    print(f"📁 Input Hybrid Dataset : {HYBRID_DATASET_DIR}")
    print(f"📁 Target Output Dir    : {DCFM_OUT_DIR}\n")

    for split in SPLITS:
        split_dir = HYBRID_DATASET_DIR / split
        img_patch_dir = split_dir / "image_patches"
        mask_patch_dir = split_dir / "mask_patches"

        if not img_patch_dir.exists() or not mask_patch_dir.exists():
            print(f"⚠️ Missing patch directories for split folder: '{split}'. Skipping...")
            continue

        split_out_img_dir = DCFM_OUT_DIR / split / "images"
        split_out_mask_dir = DCFM_OUT_DIR / split / "masks"

        if (DCFM_OUT_DIR / split).exists():
            shutil.rmtree(DCFM_OUT_DIR / split)

        split_out_img_dir.mkdir(parents=True, exist_ok=True)
        split_out_mask_dir.mkdir(parents=True, exist_ok=True)

        img_files = sorted([p for p in img_patch_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        print(f"================ CLASSIFYING PATCHES IN FOLDER [{split.upper()}] ================")
        print(f"Total patches to process: {len(img_files)}")

        dataset = PatchInferenceDataset(img_files, mask_patch_dir, vit_transform, RUST_THRESHOLD)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        y_true = []
        y_pred = []
        vit_rust_patches = []
        results = []

        for batch_imgs, batch_gt, batch_rust_counts, batch_img_paths, batch_mask_paths, batch_filenames in tqdm(dataloader, desc=f"Classifying [{split.upper()}]"):
            batch_imgs = batch_imgs.to(DEVICE)

            with torch.no_grad():
                outputs = model(batch_imgs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                preds = torch.argmax(logits, dim=1).cpu().tolist()

            gt_list = batch_gt.tolist()
            rust_counts_list = batch_rust_counts.tolist()

            for i in range(len(preds)):
                pred_label = preds[i]
                gt_label = gt_list[i]
                rust_count = rust_counts_list[i]
                filename = batch_filenames[i]
                img_path = Path(batch_img_paths[i])
                mask_path = Path(batch_mask_paths[i])

                y_true.append(gt_label)
                y_pred.append(pred_label)

                # Collect ViT-classified Rust patches (Class 1) for DCFM
                if pred_label == 1:
                    vit_rust_patches.append((filename, img_path, mask_path))

                results.append({
                    "patch_name": filename,
                    "rust_pixels": rust_count,
                    "ground_truth": gt_label,
                    "vit_prediction": pred_label
                })

        # Calculate metrics against ground truth masks
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

        # Save classification CSV log
        df = pd.DataFrame(results)
        df.to_csv(DCFM_OUT_DIR / split / "vit_evaluation_results.csv", index=False)

        # Group ViT-classified Rust patches into DCFM Co-Saliency Groups (N = 12)
        print(f"📦 Grouping {len(vit_rust_patches)} ViT-classified Rust patches into N={GROUP_SIZE} folders...")
        
        groups = {}
        for fname, src_img, src_mask in vit_rust_patches:
            if '_patch_' in fname:
                parent_id = fname.rsplit('_patch_', 1)[0]
            else:
                parent_id = fname.rsplit('_', 1)[0]

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
    print("🎉 Pipeline Complete! ViT classified patches are ready in DCFM format.")
