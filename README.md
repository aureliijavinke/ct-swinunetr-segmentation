# 3D CT Segmentation using SwinUNETR

## Overview

This repository demonstrates a 3D medical image segmentation inference workflow using SwinUNETR, MONAI and PyTorch.

## Project goal

The goal of this project is to load 3D CT volumes, apply MONAI preprocessing transforms, run SwinUNETR inference and visualize segmentation outputs.

## Technologies used

* Python
* PyTorch
* MONAI
* SwinUNETR
* NiBabel
* NumPy
* Matplotlib
* Google Colab

## Dataset

The project uses CT volumes and segmentation masks in NIfTI format (`.nii` / `.nii.gz`).

The dataset is not included in this repository.

## Model

The model is based on SwinUNETR with 1 input channel and 26 output classes.

A pretrained MONAI BTCV checkpoint is used as the backbone.

The model is not fine-tuned on the spinal CT dataset, so predictions are used only for workflow demonstration.

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run inference:

```bash
python src/swinunetr_inference.py
```

Before running, update the dataset path in the script if needed.

## Results

The script generates example visualizations in the `results/` folder, including CT slices, ground truth masks and predicted segmentation outputs.

## Limitations

This is an inference-focused demonstration, not a clinically validated segmentation model.

The prediction quality is limited because the model is not fine-tuned on the dataset used in this project.

## What this project demonstrates

This project demonstrates experience with 3D medical image data, MONAI transforms, pretrained model loading, SwinUNETR inference and segmentation result visualization.

## Author

Aurēlija Viņķe


