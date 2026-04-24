
# 3D CT Segmentation using SwinUNETR

This project demonstrates 3D medical image segmentation using a SwinUNETR model in MONAI.

The task uses CT volumes and corresponding segmentation labels. Inference is performed in Google Colab using PyTorch and a T4 GPU. A pretrained SwinUNETR backbone is loaded from the MONAI model zoo, while the final segmentation head is adapted because the number of target classes differs from the pretrained BTCV model.

## Project goal

The goal of this project is to:

- load 3D CT `.nii` / `.nii.gz` volumes
- load corresponding segmentation masks
- prepare MONAI dictionary-based transforms
- use SwinUNETR for 3D segmentation inference
- visualize image, ground truth and predicted segmentation slices
- run inference without training, as required by the task

## Technologies used

- Python
- PyTorch
- MONAI
- SwinUNETR
- NiBabel
- Matplotlib
- Google Colab
- T4 GPU

## Dataset structure

The dataset should be stored in Google Drive as:

```text
dataset (2)/
└── images/
    ├── volumes/
    │   ├── image_1.nii.gz
    │   ├── image_2.nii.gz
    │   └── ...
    └── segmentations/
        ├── label_1.nii.gz
        ├── label_2.nii.gz
        └── ...
