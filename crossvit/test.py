import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, ViTForImageClassification
import transformers.models.vit.modeling_vit as vit_module
import types
from NWRD_dataset import NWRD
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import os
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms

# -------------------------------------------------------------------
# Catch-all unpickling fix for older Hugging Face checkpoints
# -------------------------------------------------------------------
class DummyModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

class DynamicVitModule(types.ModuleType):
    def __getattr__(self, name):
        return DummyModule

vit_module.__class__ = DynamicVitModule

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
CUDA_LAUNCH_BLOCKING = 1
TORCH_USE_CUDA_DSA = 1
print("device is:", device)

transformations = transforms.Compose([
    transforms.ToTensor()            
])

test_ds = NWRD(root_dir="../data/NWRD_test", train=False, transform=transformations)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model.classifier = torch.nn.Linear(768, 2)
model.to(device)

# Load raw checkpoint
checkpoint = torch.load('./22.pth', map_location=device, weights_only=False)

if hasattr(checkpoint, 'state_dict'):
    raw_state_dict = checkpoint.state_dict()
elif isinstance(checkpoint, dict):
    raw_state_dict = checkpoint
else:
    raw_state_dict = checkpoint

# -------------------------------------------------------------------
# Remap key names from legacy Hugging Face ViT to current version
# -------------------------------------------------------------------
def remap_vit_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        new_key = new_key.replace("vit.encoder.layer.", "vit.layers.")
        new_key = new_key.replace(".attention.attention.query.", ".attention.q_proj.")
        new_key = new_key.replace(".attention.attention.key.", ".attention.k_proj.")
        new_key = new_key.replace(".attention.attention.value.", ".attention.v_proj.")
        new_key = new_key.replace(".attention.output.dense.", ".attention.o_proj.")
        new_key = new_key.replace(".intermediate.dense.", ".mlp.fc1.")
        new_key = new_key.replace(".output.dense.", ".mlp.fc2.")
        new_state_dict[new_key] = value
    return new_state_dict

converted_state_dict = remap_vit_keys(raw_state_dict)

# Load transformed weights seamlessly
model.load_state_dict(converted_state_dict)
print("Successfully mapped and loaded model weights from 22.pth!")

criterion = torch.nn.CrossEntropyLoss()

# Testing
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

rust_dir = "./results/nwrd22/rust"
fn_dir = "./results/nwrd22/fn"
fp_dir = "./results/nwrd22/fp"

os.makedirs(rust_dir, exist_ok=True)
os.makedirs(fn_dir, exist_ok=True)
os.makedirs(fp_dir, exist_ok=True)

img_paths = test_ds.images
count = 0

model.eval()
loop = tqdm(enumerate(test_loader), total=len(test_loader))
with torch.no_grad():
    for batch_idx, (images, labels) in loop:
        inputs = processor(images=images, return_tensors="pt", do_rescale=False).to(device)
        labels = labels.to(device)

        outputs = model(**inputs)
        logits = outputs.logits
        prediction = logits.argmax(axis=1)

        filename = os.path.basename(img_paths[count])

        if (prediction == 1):
            image_path = os.path.join(rust_dir, filename)
            image = images.squeeze().cpu()
            save_image(image, image_path)

        if ((prediction == 1) and (labels == 0)):
            image_path = os.path.join(fp_dir, filename)
            image = images.squeeze().cpu()
            save_image(image, image_path)

        if ((prediction == 0) and (labels == 1)):
            image_path = os.path.join(fn_dir, filename)
            image = images.squeeze().cpu()
            save_image(image, image_path)

        true_positives += torch.sum((prediction == 1) & (labels == 1)).item()
        false_positives += torch.sum((prediction == 1) & (labels == 0)).item()
        true_negatives += torch.sum((prediction == 0) & (labels == 0)).item()
        false_negatives += torch.sum((prediction == 0) & (labels == 1)).item()
        count += 1

# Calculate metrics
precision = true_positives / (true_positives + false_positives + 1e-10)
recall = true_positives / (true_positives + false_negatives + 1e-10)
F1 = 2 * (precision * recall) / (precision + recall)
accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives + 1e-10)

# Print metrics
print(f"\nTrue Positives: {true_positives}")
print(f"False Positives: {false_positives}")
print(f"True Negatives: {true_negatives}")
print(f"False Negatives: {false_negatives}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {F1:.4f}")
print(f"Accuracy: {accuracy:.4f}")