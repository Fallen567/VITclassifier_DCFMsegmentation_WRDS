import os
import cv2
import numpy as np
from tqdm import tqdm

pred_base = r"D:\downloads_v2\academic docs\tukl_internship\codefiles\DCFM_ViT\Co-salient-feature-extraction-for-WRD\DCFM-master\CoSODmaps\pred\NWRD\nwrd22\rust"
gt_base   = r"..\data\NWRD_test\gt_patches_exact\rust"

# Build dictionary mapping clean stem names (e.g. "25_patch_5") -> full path
def get_file_dict(folder):
    file_dict = {}
    for f in os.listdir(folder):
        if f.startswith('heatmap') or not f.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        # Strip extension and common prefixes like 'pred_' or 'mask_'
        stem = os.path.splitext(f)[0].replace('pred_', '').replace('mask_', '')
        file_dict[stem] = os.path.join(folder, f)
    return file_dict

pred_dict = get_file_dict(pred_base)
gt_dict   = get_file_dict(gt_base)

# Find all unique stems across both GT and Predictions
all_stems = sorted(list(set(gt_dict.keys()).union(set(pred_dict.keys()))))

print(f"Total Unique Patches to Evaluate: {len(all_stems)}")
print(f"GT Patches Available            : {len(gt_dict)}")
print(f"Prediction Patches Available    : {len(pred_dict)}\n")

gt_pos_total = 0
pred_pos_total = np.zeros(256, dtype=np.float64)
tp_total = np.zeros(256, dtype=np.float64)
total_mae = 0.0
count = 0

for stem in tqdm(all_stems, desc="Evaluating All Patches"):
    g_path = gt_dict.get(stem)
    p_path = pred_dict.get(stem)

    # 1. Load GT (if missing, treat as all-zero background)
    if g_path and os.path.exists(g_path):
        gt = cv2.imread(g_path, cv2.IMREAD_GRAYSCALE)
    else:
        gt = np.zeros((224, 224), dtype=np.uint8)

    # 2. Load Prediction (if missing, treat as all-zero prediction)
    if p_path and os.path.exists(p_path):
        pred = cv2.imread(p_path, cv2.IMREAD_GRAYSCALE)
    else:
        pred = np.zeros((224, 224), dtype=np.uint8)

    if pred is None or gt is None:
        continue

    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

    # GT binary mask
    gt_bin = (gt >= 128).astype(np.uint8)

    # MAE
    pred_norm = pred.astype(np.float32) / 255.0
    total_mae += np.mean(np.abs(pred_norm - gt_bin))
    count += 1

    # GT Positives
    gt_pos = np.sum(gt_bin == 1)
    gt_pos_total += gt_pos

    # Fast Vectorized Histogram Accumulation
    hist_pred = np.bincount(pred.ravel(), minlength=256)
    hist_tp   = np.bincount(pred[gt_bin == 1].ravel(), minlength=256)

    pred_pos_total += np.cumsum(hist_pred[::-1])[::-1]
    tp_total       += np.cumsum(hist_tp[::-1])[::-1]

# Compute Precision, Recall across thresholds
precisions = tp_total[1:255] / (pred_pos_total[1:255] + 1e-8)
recalls    = tp_total[1:255] / (gt_pos_total + 1e-8)

# Paper Standard Weighted F-beta (beta^2 = 0.3)
beta_sq = 0.3
f_beta_measures = ((1 + beta_sq) * precisions * recalls) / (beta_sq * precisions + recalls + 1e-8)

# Standard Harmonic F1 (beta^2 = 1.0)
f1_standard = (2 * precisions * recalls) / (precisions + recalls + 1e-8)

# Fixed Threshold @ T = 128 (Paper Standard)
fixed_idx = 127  # T = 128

print("\n" + "="*55)
print("       DCFM RUST FULL EVALUATION (ALL PATCHES)       ")
print("="*55)
print(f" Total Evaluated Patches : {count}")
print(f" MAE                     : {total_mae / count:.4f}")
print("-" * 55)
print(" [PAPER BENCHMARK RESULTS @ FIXED THRESHOLD T = 128]")
print(f" F_beta Score (beta^2=0.3): {f_beta_measures[fixed_idx]:.4f}  <-- PAPER METRIC")
print(f" Standard F1 (beta^2=1.0) : {f1_standard[fixed_idx]:.4f}")
print(f" Precision               : {precisions[fixed_idx]:.4f}")
print(f" Recall                  : {recalls[fixed_idx]:.4f}")
print("="*55)