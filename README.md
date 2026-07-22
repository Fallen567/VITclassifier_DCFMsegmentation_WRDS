<<<<<<< HEAD
# Co-Salient Feature Extraction for Wheat Rust Disease Segmentation

This repository contains the code and resources for the paper:

> **Segmentation of Wheat Rust Disease Using Co-Salient Feature Extraction**
> Anwar H., Muhammad H., Ghaffar M. M., Afridi M. A., Khan M. J., Weis C., Wehn N., Shafait F.
> *AgriEngineering*, 2025, 7(2), 23.
> DOI: [10.3390/agriengineering7020023](https://doi.org/10.3390/agriengineering7020023)

**Code and repository maintained by [Haseeb Muhammad](https://github.com/Haseeb-Muhammad-ekai).**

## Overview

Semantic segmentation of wheat yellow/stripe rust disease (WRD) images to classify pixels as rust-affected or healthy. The pipeline uses a two-stage approach:

1. **ViT Classifier** — Binary Vision Transformer classifies 224×224 patches as rust or non-rust.
2. **Co-SOD Segmentation (DCFM)** — Groups of 12 classified patches are fed to a Co-Salient Object Detection model (DCFM) for co-salient feature extraction and pixel-level segmentation.

This achieves higher segmentation performance with **5× less training time** compared to prior work.

## Results

| Metric    | Rust Class |
|-----------|-----------|
| F1 Score  | 0.638     |
| Precision | 0.621     |
| Recall    | 0.675     |

## Dataset

The **NWRD** (Nathiagali Wheat Rust Disease) dataset is a real-world segmentation dataset of wheat rust diseased and healthy leaf images, specifically constructed for semantic segmentation of wheat rust disease. It consists of 100 annotated images with binary masks.

Sample images from NWRD — rust disease regions with binary masks:

![image](https://github.com/user-attachments/assets/3e7eff0e-f546-4673-ac21-3c199af80628)

Dataset available at: https://dll.seecs.nust.edu.pk/downloads/

## Environment Setup

```bash
pip install -r requirements.txt
```

## Data Format

Place datasets under `DCFM/data/` as follows:

```
Co-salient-feature-extraction-for-WRD/
├── other codes
├── ...
└── data/
    ├── NWRD_train/
    ├── NWRD_val/
    └── NWRD_test/
```

## Trained Model

Download pretrained weights: [Google Drive](https://drive.google.com/drive/folders/1kvPTjDiOU6_puIWmNVoKYcoLIxuSIz82?usp=sharing)

Run inference:

```bash
python test.py
```

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{anwar2025wheat,
  title={Segmentation of Wheat Rust Disease Using Co-Salient Feature Extraction},
  author={Anwar, H. and Muhammad, H. and Ghaffar, M. M. and Afridi, M. A. and Khan, M. J. and Weis, C. and Wehn, N. and Shafait, F.},
  journal={AgriEngineering},
  volume={7},
  number={2},
  pages={23},
  year={2025},
  publisher={MDPI},
  doi={10.3390/agriengineering7020023}
}
```

## Acknowledgements

Code builds on [DCFM](https://github.com/siyueyu/DCFM).
=======
# VITclassifier_DCFMsegmentation_WRDS
>>>>>>> 28f27fd0faaa378c89de5d5a47d7ba2d8f3e5c19
