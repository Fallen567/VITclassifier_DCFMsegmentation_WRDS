import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor
import wandb

# ==========================================
# 1. DATASET DEFINITION
# ==========================================
class HybridPatchDataset(Dataset):
    def __init__(self, img_dir, mask_dir, processor, threshold=150):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.threshold = threshold

        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
        self.img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(valid_exts)
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")

        stem = os.path.splitext(img_name)[0]
        # Remove augmentation suffixes if present to match ground-truth mask stem
        for suffix in ["_orig", "_hflip", "_vflip", "_rot90", "_rot45", "_rot180", "_rot135"]:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break

        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            candidate = os.path.join(self.mask_dir, stem + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break

        if mask_path is not None:
            mask = Image.open(mask_path).convert("L")
            rust_pixels = np.sum(np.array(mask) > 128)
            label = 1 if rust_pixels >= self.threshold else 0
        else:
            label = 0

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.squeeze(0)

        return pixel_values, torch.tensor(label, dtype=torch.long)

# ==========================================
# 2. MAIN TRAINING FUNCTION
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Train ViT Classifier on Hybrid Wheat Rust Dataset")
    parser.add_argument("--dataset_dir", type=str, 
                        default=r"D:\downloads_v2\academic docs\tukl_internship\dataset\FRDIxNWRD_hybrid",
                        help="Path to hybrid dataset root directory")
    # PAPER HYPERPARAMETERS
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (Paper: 50)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (Paper: 8)")
    parser.add_argument("--lr", type=float, default=3e-8, help="Learning rate (Paper: 3e-8)")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay (Paper: 0.001)")
    
    # SYSTEM & WANDB
    parser.add_argument("--save_dir", type=str, default="./checkpoints_vit", help="Checkpoint save directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pth to resume training")
    parser.add_argument("--wandb_project", type=str, default="crossvit_rust_classifier_new", help="W&B Project Name")
    parser.add_argument("--wandb_key", type=str, default="", help="Optional W&B API Key")
    args = parser.parse_args()

    # Authenticate W&B if key provided
    if args.wandb_key:
        wandb.login(key=args.wandb_key)

    # Reproducibility settings
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚡ Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Load Pretrained ViT Base Model
    model_name = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    model.classifier = nn.Linear(model.config.hidden_size, 2)
    model.to(device)

    # 2. Setup Data Loaders
    train_img_dir = os.path.join(args.dataset_dir, "train", "augmented_image_patches")
    train_mask_dir = os.path.join(args.dataset_dir, "train", "augmented_mask_patches")
    
    val_img_dir = os.path.join(args.dataset_dir, "val", "image_patches")
    val_mask_dir = os.path.join(args.dataset_dir, "val", "mask_patches")

    train_ds = HybridPatchDataset(train_img_dir, train_mask_dir, processor)
    val_ds = HybridPatchDataset(val_img_dir, val_mask_dir, processor)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"📊 Training patches: {len(train_ds)} | Validation patches: {len(val_ds)}")

    # 3. Paper-Exact Optimizer & Loss
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # 4. Resume Checkpoint Handling (Power Outage Recovery)
    start_epoch = 0
    best_val_acc = 0.0

    if args.resume and os.path.exists(args.resume):
        print(f"🔄 Resuming checkpoint from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"🟢 Resumed successfully. Starting at epoch {start_epoch}")

    # 5. Initialize W&B Run
    wandb.init(project=args.wandb_project, config=vars(args), resume="allow")

    # 6. Training Loop
    for epoch in range(start_epoch, args.epochs):
        # --- TRAIN ---
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=images)
            logits = outputs.logits

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            loop.set_postfix(loss=f"{np.mean(train_losses):.4f}", acc=f"{(train_correct/train_total)*100:.2f}%")

        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = (train_correct / train_total) * 100

        # --- VALIDATION ---
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for images, labels in val_loop:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(pixel_values=images)
                logits = outputs.logits

                loss = criterion(logits, labels)
                val_losses.append(loss.item())

                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                val_loop.set_postfix(loss=f"{np.mean(val_losses):.4f}", acc=f"{(val_correct/val_total)*100:.2f}%")

        epoch_val_loss = np.mean(val_losses)
        epoch_val_acc = (val_correct / val_total) * 100

        # Log metrics to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "val_acc": epoch_val_acc
        })

        # --- SAVE CHECKPOINTS ---
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'val_loss': epoch_val_loss
        }

        # 1. Latest checkpoint for crash recovery
        latest_path = os.path.join(args.save_dir, "latest_checkpoint.pth")
        torch.save(checkpoint_data, latest_path)

        # 2. Save best model based on validation accuracy
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            checkpoint_data['best_val_acc'] = best_val_acc
            best_path = os.path.join(args.save_dir, "best_vit_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"⭐ Best model saved with Val Acc: {best_val_acc:.2f}%")

    print("🎉 Training Complete!")

if __name__ == "__main__":
    main()