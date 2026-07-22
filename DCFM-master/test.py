# from PIL import Image
from dataset import get_loader
import torch
from torchvision import transforms
from util import save_tensor_img, Logger
from tqdm import tqdm
from torch import nn
import os
import cv2
import numpy as np
from models.main import *
import argparse


def main(args):
    # Init model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCFM()
    model = model.to(device)
    
    try:
        modelname = "../best_ep12_Smeasure0.7256.pth"
        dcfmnet_dict = torch.load(modelname, map_location=device)
        print('Loaded checkpoint:', modelname)
    except Exception as e:
        print(f"Failed to load relative checkpoint, falling back to param_root: {e}")
        fallback_path = os.path.join(args.param_root, 'dcfm.pth')
        dcfmnet_dict = torch.load(fallback_path, map_location=device)
        print('Loaded checkpoint:', fallback_path)

    model.dcfmnet.load_state_dict(dcfmnet_dict)
    model.eval()
    model.set_mode('test')

    for testset in ['NWRD']:
        if testset == 'NWRD':
            test_img_path = '../crossvit/results/nwrd22/'
            test_gt_path = '../crossvit/results/nwrd22/'
            saved_root = os.path.join(args.save_root, 'NWRD')
        else:
            print('Unknown test dataset:', testset)
            continue

        test_loader = get_loader(
            test_img_path, test_gt_path, args.size, 1, istrain=False, shuffle=False, num_workers=0, pin=False
        )

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Segmenting Subfolders"):
                subpaths = batch[2]
                
                # Standardize path separators
                sample_subpath = subpaths[0][0].replace('\\', '/')
                
                # FLEXIBLE FILTER: Process if 'rust' is anywhere in the folder path
                if 'rust' not in sample_subpath.lower():
                    continue

                inputs = batch[0].to(device).squeeze(0)
                gts = batch[1].to(device).squeeze(0)
                ori_sizes = batch[3]

                # Chunking mechanism: Process rust patches in micro-groups of 5
                total_imgs = inputs.shape[0]
                CHUNK_SIZE = 5
                
                scaled_preds_list = []
                for start_idx in range(0, total_imgs, CHUNK_SIZE):
                    end_idx = min(start_idx + CHUNK_SIZE, total_imgs)
                    chunk_inputs = inputs[start_idx:end_idx]
                    chunk_gts = gts[start_idx:end_idx]
                    
                    chunk_preds = model(chunk_inputs, chunk_gts)
                    chunk_preds = torch.sigmoid(chunk_preds[-1])
                    scaled_preds_list.append(chunk_preds)
                
                scaled_preds = torch.cat(scaled_preds_list, dim=0)

                num = gts.shape[0]
                for inum in range(num):
                    subpath = subpaths[inum][0].replace('\\', '/')
                    parent_dir = os.path.dirname(subpath)
                    
                    target_dir = os.path.join(saved_root, parent_dir) if parent_dir else saved_root
                    os.makedirs(target_dir, exist_ok=True)

                    heatmap_dir = os.path.join(saved_root, 'heatmaps', parent_dir) if parent_dir else os.path.join(saved_root, 'heatmaps')
                    os.makedirs(heatmap_dir, exist_ok=True)

                    ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                    res = nn.functional.interpolate(
                        scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear', align_corners=True
                    )
                    
                    filename = os.path.basename(subpath)
                    
                    # 1. Save standard prediction mask
                    save_tensor_img(res, os.path.join(target_dir, filename))

                    # 2. Save colorized JET Heatmap
                    pred_np = res.squeeze().cpu().numpy()
                    pred_uint8 = (pred_np * 255).astype(np.uint8)
                    heatmap_colored = cv2.applyColorMap(pred_uint8, cv2.COLORMAP_JET)

                    cv2.imwrite(os.path.join(heatmap_dir, f"heatmap_{filename}"), heatmap_colored)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCFM Co-salient Feature Extraction')
    parser.add_argument('--size', default=224, type=int, help='input size')
    parser.add_argument('--param_root', default='/data1/dcfm/temp', type=str, help='model folder')
    parser.add_argument('--save_root', default='./CoSODmaps/pred', type=str, help='Output folder')

    args = parser.parse_args()
    main(args)