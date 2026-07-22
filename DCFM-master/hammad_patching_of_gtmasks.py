import os
import cv2
import numpy as np
from tqdm import tqdm

pred_patch_dir = r"D:\downloads_v2\academic docs\tukl_internship\codefiles\DCFM_ViT\Co-salient-feature-extraction-for-WRD\DCFM-master\CoSODmaps\pred\NWRD\nwrd22\rust"
full_gt_dir    = r"D:\downloads_v2\academic docs\tukl_internship\codefiles\DCFM_ViT\Co-salient-feature-extraction-for-WRD\data\NWRD_test\masks"
output_gt_patches = r"D:\downloads_v2\academic docs\tukl_internship\codefiles\DCFM_ViT\Co-salient-feature-extraction-for-WRD\data\NWRD_test\gt_patches\rust"

os.makedirs(output_gt_patches, exist_ok=True)

patches = [f for f in os.listdir(pred_patch_dir) if f.endswith(('.png', '.jpg'))]

# Group predicted patches by original image ID (e.g. '105' from '105_patch_1.png')
image_groups = {}
for p in patches:
    img_id = p.split('_patch_')[0]
    image_groups.setdefault(img_id, []).append(p)

print(f"Cropping Ground Truth masks into patches for {len(image_groups)} images...")

total_cropped = 0

for img_id, patch_files in tqdm(image_groups.items(), desc="Cropping GT Masks"):
    gt_path = os.path.join(full_gt_dir, f"{img_id}.png")
    if not os.path.exists(gt_path):
        gt_path = os.path.join(full_gt_dir, f"{img_id}.jpg")
    if not os.path.exists(gt_path):
        print(f"Warning: Full GT mask for image '{img_id}' not found. Skipping.")
        continue

    gt_full = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt_full is None:
        continue

    H, W = gt_full.shape
    num_patches = len(patch_files)
    grid_size = int(np.ceil(np.sqrt(num_patches)))
    p_h, p_w = H // grid_size, W // grid_size

    # Sort patches by patch number numerical index
    patch_files.sort(key=lambda x: int(x.split('_patch_')[1].split('.')[0]) if '_patch_' in x and x.split('_patch_')[1].split('.')[0].isdigit() else x)

    for idx, p_file in enumerate(patch_files):
        # Crop region from full GT
        r, c = idx // grid_size, idx % grid_size
        y1, y2 = r * p_h, (r + 1) * p_h if r < grid_size - 1 else H
        x1, x2 = c * p_w, (c + 1) * p_w if c < grid_size - 1 else W

        gt_patch = gt_full[y1:y2, x1:x2]

        # Standardize GT to clean binary 0 or 255
        _, gt_patch_bin = cv2.threshold(gt_patch, 128, 255, cv2.THRESH_BINARY)

        # Save with exact same filename as the predicted patch
        out_path = os.path.join(output_gt_patches, p_file)
        cv2.imwrite(out_path, gt_patch_bin)
        total_cropped += 1

print(f"\nDone! Successfully generated {total_cropped} GT patches in:\n{output_gt_patches}")