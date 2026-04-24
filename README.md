
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

Model
The model used is:
SwinUNETR(
    in_channels=1,
    out_channels=26,
    feature_size=48,
    use_checkpoint=True
)

The output has 26 classes:

background
25 vertebrae labels

A pretrained MONAI SwinUNETR BTCV checkpoint is loaded. The old output segmentation head is removed before loading weights because the number of output classes is different.

Important note
Training is skipped. The model performs inference using a pretrained backbone only.

Prediction quality is expected to be limited because the model is not fine-tuned on this specific spinal CT dataset.

How to run
Install dependencies:
pip install -r requirements.txt

Run in Google Colab:
python src/swinunetr_inference.py

Or open the notebook version in:
notebooks/swinunetr_inference_colab.ipynb

Results

The script visualizes:

CT image slice
ground truth segmentation
predicted segmentation
binary label comparison
nearby slice comparison

Example output figures should be saved in the results/ folder.

Author
Aurēlija
