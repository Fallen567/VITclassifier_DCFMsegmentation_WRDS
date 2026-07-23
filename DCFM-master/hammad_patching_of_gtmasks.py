import os
import cv2
import glob

# Set your paths based on your workspace
full_masks_dir = r"..\data\NWRD_test\masks"
rust_patches_dir = r"..\data\NWRD_test\rust"
output_gt_dir = r"..\data\NWRD_test\gt_patches_exact\rust"

os.makedirs(output_gt_dir, exist_ok=True)

# Grid parameters (standard 224x224 patch grid used during ViT cropping)
PATCH_SIZE = 224  

# Get all rust patch names
patch_paths = glob.glob(os.path.join(rust_patches_dir, "*.png")) + glob.glob(os.path.join(rust_patches_dir, "*.jpg"))

print(f"Found {len(patch_paths)} rust patches. Cropping matching GT mask patches...")

for patch_path in patch_paths:
    patch_filename = os.path.basename(patch_path) # e.g. "25_patch_5.png"
    
    # Extract base image ID (e.g., "25") and patch index (e.g., "5")
    parts = patch_filename.replace('.png', '').replace('.jpg', '').split('_patch_')
    if len(parts) < 2:
        continue
        
    img_id, patch_num = parts[0], int(parts[1])
    
    # Find full GT mask (e.g., "25.png")
    full_mask_path = os.path.join(full_masks_dir, f"{img_id}.png")
    if not os.path.exists(full_mask_path):
        full_mask_path = os.path.join(full_masks_dir, f"{img_id}.jpg")
        
    if not os.path.exists(full_mask_path):
        print(f"Warning: Full mask for {img_id} not found at {full_mask_path}")
        continue
        
    # Read full mask
    full_mask = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)
    h, w = full_mask.shape
    
    # Calculate grid position based on patch number
    # Assuming standard left-to-right, top-to-bottom grid slicing
    cols = w // PATCH_SIZE
    row = patch_num // cols
    col = patch_num % cols
    
    y1, y2 = row * PATCH_SIZE, (row + 1) * PATCH_SIZE
    x1, x2 = col * PATCH_SIZE, (col + 1) * PATCH_SIZE
    
    # Crop GT patch
    gt_patch = full_mask[y1:y2, x1:x2]
    
    # Ensure dimensions match patch size exactly
    if gt_patch.shape[0] != PATCH_SIZE or gt_patch.shape[1] != PATCH_SIZE:
        gt_patch = cv2.resize(gt_patch, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_NEAREST)
        
    # Save cropped GT patch
    save_path = os.path.join(output_gt_dir, patch_filename)
    cv2.imwrite(save_path, gt_patch)

print(f"Done! Saved GT patches to {output_gt_dir}")