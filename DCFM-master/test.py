# from PIL import Image
from dataset import get_loader
import torch
from torchvision import transforms
from util import save_tensor_img, Logger
from tqdm import tqdm
from torch import nn
import os
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
        if testset == 'CoCA':
            test_img_path = './data/images/CoCA/'
            test_gt_path = './data/gts/CoCA/'
            saved_root = os.path.join(args.save_root, 'CoCA')
        elif testset == 'CoSOD3k':
            test_img_path = './data/images/CoSOD3k/'
            test_gt_path = './data/gts/CoSOD3k/'
            saved_root = os.path.join(args.save_root, 'CoSOD3k')
        elif testset == 'CoSal2015':
            test_img_path = './data/images/CoSal2015/'
            test_gt_path = './data/gts/CoSal2015/'
            saved_root = os.path.join(args.save_root, 'CoSal2015')
        elif testset == 'NWRD':
            # Point to parent directory so DCFM recognizes 'rust' as the category folder
            test_img_path = '../crossvit/results/nwrd22/'
            test_gt_path = '../crossvit/results/nwrd22/'
            saved_root = os.path.join(args.save_root, 'NWRD')
        else:
            print('Unknown test dataset:', args.dataset)
            continue

        test_loader = get_loader(
            test_img_path, test_gt_path, args.size, 1, istrain=False, shuffle=False, num_workers=2, pin=True
        )

        for batch in tqdm(test_loader):
            inputs = batch[0].to(device).squeeze(0)
            gts = batch[1].to(device).squeeze(0)
            subpaths = batch[2]
            ori_sizes = batch[3]

            scaled_preds = model(inputs, gts)
            scaled_preds = torch.sigmoid(scaled_preds[-1])

            num = gts.shape[0]
            for inum in range(num):
                # Standardize path separators for Windows
                subpath = subpaths[inum][0].replace('\\', '/')
                
                # Determine relative path structure
                parent_dir = os.path.dirname(subpath)
                target_dir = os.path.join(saved_root, parent_dir) if parent_dir else saved_root
                os.makedirs(target_dir, exist_ok=True)

                ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                res = nn.functional.interpolate(
                    scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear', align_corners=True
                )
                save_tensor_img(res, os.path.join(saved_root, subpath))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCFM Co-salient Feature Extraction')
    parser.add_argument('--size', default=224, type=int, help='input size')
    parser.add_argument('--param_root', default='/data1/dcfm/temp', type=str, help='model folder')
    parser.add_argument('--save_root', default='./CoSODmaps/pred', type=str, help='Output folder')

    args = parser.parse_args()
    main(args)