import os

IMG_DIR = r"D:\downloads_v2\academic docs\tukl_internship\dataset\FRDIxNWRD_hybrid\train\augmented_image_patches"

# Suffixes added exclusively to augmented Rust patches
RUST_SUFFIXES = ("_orig", "_hflip", "_vflip", "_rot90", "_rot45", "_rot180", "_rot135")

valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
all_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(valid_exts)]

rust_count = 0
non_rust_count = 0

for filename in all_files:
    stem = os.path.splitext(filename)[0]
    if stem.endswith(RUST_SUFFIXES):
        rust_count += 1
    else:
        non_rust_count += 1

total_count = len(all_files)
rust_percentage = (rust_count / total_count) * 100 if total_count > 0 else 0
non_rust_percentage = (non_rust_count / total_count) * 100 if total_count > 0 else 0

print("=" * 50)
print("📊 FINAL AUGMENTED DATASET COUNTS")
print("=" * 50)
print(f"📁 Folder: {IMG_DIR}")
print(f"🟢 Total Rust Patches:     {rust_count:,} ({rust_percentage:.2f}%)")
print(f"⚪ Total Non-Rust Patches: {non_rust_count:,} ({non_rust_percentage:.2f}%)")
print(f"📦 Total Combined Patches: {total_count:,}")
print("=" * 50)