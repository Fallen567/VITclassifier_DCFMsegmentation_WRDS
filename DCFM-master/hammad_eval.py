import os
import cv2
import numpy as np
from tqdm import tqdm

pred_base = r"D:\downloads_v2\academic docs\tukl_internship\codefiles\DCFM_ViT\Co-salient-feature-extraction-for-WRD\DCFM-master\CoSODmaps\pred\NWRD\nwrd22\rust"
gt_base   = r"D:\downloads_v2\academic docs\tukl_internship\codefiles\DCFM_ViT\Co-salient-feature-extraction-for-WRD\data\NWRD_test\gt_patches\rust"

pred_files = [f for f in os.listdir(pred_base) if f.endswith(('.png', '.jpg', '.bmp')) and not f.startswith('heatmap')]
gt_files = set(os.listdir(gt_base))

matched_files = [f for f in pred_files if f in gt_files]

print(f"Evaluating {len(matched_files)} matched patch masks...\n")

gt_pos_total = 0
pred_pos_total = np.zeros(256, dtype=np.float64)
tp_total = np.zeros(256, dtype=np.float64)
total_mae = 0.0
count = 0

for fname in tqdm(matched_files, desc="Processing Masks"):
    p_path = os.path.join(pred_base, fname)
    g_path = os.path.join(gt_base, fname)

    pred = cv2.imread(p_path, cv2.IMREAD_GRAYSCALE)
    gt   = cv2.imread(g_path, cv2.IMREAD_GRAYSCALE)

    if pred is None or gt is None:
        continue

    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

    # GT is binary 0 or 1
    gt_bin = (gt >= 128).astype(np.uint8)

    # MAE
    pred_norm = pred.astype(np.float32) / 255.0
    total_mae += np.mean(np.abs(pred_norm - gt_bin))
    count += 1

    # GT Positives
    gt_pos = np.sum(gt_bin == 1)
    gt_pos_total += gt_pos

    # Fast Vectorized Histogram Accumulation across 255 thresholds
    hist_pred = np.bincount(pred.ravel(), minlength=256)
    hist_tp   = np.bincount(pred[gt_bin == 1].ravel(), minlength=256)

    # Reverse cumulative sum gets counts >= threshold t
    pred_pos_total += np.cumsum(hist_pred[::-1])[::-1]
    tp_total       += np.cumsum(hist_tp[::-1])[::-1]

# Compute Precision, Recall, and Weighted F-beta (beta^2 = 0.3)
precisions = tp_total[1:255] / (pred_pos_total[1:255] + 1e-8)
recalls    = tp_total[1:255] / (gt_pos_total + 1e-8)

beta_sq = 0.3
f_measures = ((1 + beta_sq) * precisions * recalls) / (beta_sq * precisions + recalls + 1e-8)

best_idx = np.argmax(f_measures)
best_t = best_idx + 1  # Threshold 1 to 254

print("\n" + "="*45)
print("     DCFM RUST SEGMENTATION RESULTS (PATCHES)   ")
print("="*45)
print(f" Total Matched Masks   : {count}")
print(f" Optimal Threshold     : {best_t} / 255")
print("-" * 45)
print(f" Max F-measure (F_max) : {f_measures[best_idx]:.4f}")
print(f" Precision @ F_max     : {precisions[best_idx]:.4f}")
print(f" Recall @ F_max        : {recalls[best_idx]:.4f}")
print(f" MAE                   : {total_mae / count:.4f}")
print("="*45)