import os
import glob
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
)

from monai.data import Dataset
from monai.networks.nets import SwinUNETR
from monai.bundle import download


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "/content/drive/MyDrive/dataset (2)/images"

VOLUME_DIR = os.path.join(DATASET_PATH, "volumes")
SEG_DIR_1 = os.path.join(DATASET_PATH, "segmentations")
SEG_DIR_2 = os.path.join(DATASET_PATH, "segmentation")

NUM_CLASSES = 26
CROP_SIZE = (64, 64, 64)


# ============================================================
# DATASET PATH CHECK
# ============================================================

if os.path.exists(SEG_DIR_1):
    SEG_DIR = SEG_DIR_1
elif os.path.exists(SEG_DIR_2):
    SEG_DIR = SEG_DIR_2
else:
    raise FileNotFoundError(
        f"Segmentation folder not found.\nChecked:\n{SEG_DIR_1}\n{SEG_DIR_2}"
    )

image_paths = sorted(glob.glob(os.path.join(VOLUME_DIR, "*.nii*")))
mask_paths = sorted(glob.glob(os.path.join(SEG_DIR, "*.nii*")))

print("Volume folder:", VOLUME_DIR)
print("Segmentation folder:", SEG_DIR)
print("Images found:", len(image_paths))
print("Masks found:", len(mask_paths))

if len(image_paths) == 0:
    raise ValueError("No image files found in the volumes folder.")

if len(mask_paths) == 0:
    raise ValueError("No mask files found in the segmentation folder.")

if len(image_paths) != len(mask_paths):
    raise ValueError(
        f"Mismatch between images ({len(image_paths)}) and masks ({len(mask_paths)})."
    )

print("\nImage / mask pairs:")
for i, (img, msk) in enumerate(zip(image_paths, mask_paths)):
    print(f"{i}: {os.path.basename(img)} <-> {os.path.basename(msk)}")


# ============================================================
# MASK CONTENT CHECK
# ============================================================

print("\nMask content check:")
for p in mask_paths:
    arr = nib.load(p).get_fdata()
    uniq = np.unique(arr)

    print(
        os.path.basename(p),
        "| shape:", arr.shape,
        "| min:", arr.min(),
        "| max:", arr.max(),
        "| unique(sample):", uniq[:10],
    )


# ============================================================
# MONAI DATASET
# ============================================================

data = [{"image": img, "label": seg} for img, seg in zip(image_paths, mask_paths)]

transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=CROP_SIZE,
            num_samples=1,
            pos=1,
            neg=1,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    ]
)

train_ds = Dataset(data=data, transform=transforms)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nUsing device:", device)

if device.type != "cuda":
    print("WARNING: GPU is not enabled. In Colab choose Runtime -> Change runtime type -> T4 GPU.")


# ============================================================
# MODEL
# ============================================================

model = SwinUNETR(
    in_channels=1,
    out_channels=NUM_CLASSES,
    feature_size=48,
    use_checkpoint=True,
).to(device)

print("\nDownloading MONAI pretrained SwinUNETR bundle...")
download(name="swin_unetr_btcv_segmentation", bundle_dir="./models")

ckpt_path = "./models/swin_unetr_btcv_segmentation/models/model.pt"
checkpoint = torch.load(ckpt_path, map_location=device)

state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

# Remove old segmentation head because output classes differ.
state_dict = {k: v for k, v in state_dict.items() if "out.conv" not in k}

model.load_state_dict(state_dict, strict=False)
print("Backbone weights loaded successfully.")

# Freeze backbone.
for name, param in model.named_parameters():
    if "out" not in name:
        param.requires_grad = False

model.eval()


# ============================================================
# SELECT CROP WITH NON-EMPTY MASK
# ============================================================

selected_sample = None
selected_case_index = None
selected_crop_index = None

with torch.no_grad():
    for case_idx in range(len(train_ds)):
        sample_list = train_ds[case_idx]

        for crop_idx, sample in enumerate(sample_list):
            gt = sample["label"][0].cpu().numpy()

            if gt.max() > 0:
                selected_sample = sample
                selected_case_index = case_idx
                selected_crop_index = crop_idx
                break

        if selected_sample is not None:
            break

if selected_sample is None:
    print("\nWARNING: No crop with positive mask found. Falling back to first crop of first case.")
    sample_list = train_ds[0]
    selected_sample = sample_list[0]
    selected_case_index = 0
    selected_crop_index = 0

print(f"\nSelected case index: {selected_case_index}")
print(f"Selected crop index: {selected_crop_index}")


# ============================================================
# INFERENCE
# ============================================================

with torch.no_grad():
    image = selected_sample["image"].unsqueeze(0).to(device)
    pred = model(image)
    pred = torch.argmax(pred, dim=1)

print("Prediction shape:", pred.shape)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

img = selected_sample["image"][0].cpu().numpy()
gt = selected_sample["label"][0].cpu().numpy()
pr = pred[0].cpu().numpy()

print("Image shape:", img.shape)
print("Ground truth max label:", gt.max())
print("Prediction unique labels:", np.unique(pr)[:20])


# ============================================================
# BEST SLICE SELECTION
# ============================================================

gt_voxel_counts = [(gt[:, :, z] > 0).sum() for z in range(gt.shape[2])]
pr_voxel_counts = [(pr[:, :, z] > 0).sum() for z in range(pr.shape[2])]

if max(gt_voxel_counts) > 0:
    best_z = int(np.argmax(gt_voxel_counts))
    slice_source = "ground truth"
elif max(pr_voxel_counts) > 0:
    best_z = int(np.argmax(pr_voxel_counts))
    slice_source = "prediction"
else:
    best_z = img.shape[2] // 2
    slice_source = "middle slice"

print("Best slice index:", best_z, f"(chosen from {slice_source})")


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Image")
plt.imshow(img[:, :, best_z], cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Ground Truth")
plt.imshow(gt[:, :, best_z])
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Prediction")
plt.imshow(pr[:, :, best_z])
plt.axis("off")

plt.tight_layout()
plt.savefig("results/multiclass_visualization.png", dpi=150)
plt.show()


# ============================================================
# ONE-LABEL BINARY VISUALIZATION
# ============================================================

gt_slice = gt[:, :, best_z]
pred_slice = pr[:, :, best_z]

nonzero_labels = gt_slice[gt_slice > 0]

if len(nonzero_labels) > 0:
    target_label = int(np.bincount(nonzero_labels.astype(int)).argmax())
else:
    target_label = 1

gt_bin = (gt_slice == target_label).astype(np.uint8)
pred_bin = (pred_slice == target_label).astype(np.uint8)

plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.title("Image")
plt.imshow(img[:, :, best_z], cmap="gray")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title(f"GT label {target_label}")
plt.imshow(gt_bin, cmap="gray")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title(f"Pred label {target_label}")
plt.imshow(pred_bin, cmap="gray")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("Overlay")
plt.imshow(img[:, :, best_z], cmap="gray")
plt.imshow(gt_bin, alpha=0.4)
plt.axis("off")

plt.tight_layout()
plt.savefig("results/binary_label_overlay.png", dpi=150)
plt.show()


# ============================================================
# NEARBY SLICES
# ============================================================

candidate_slices = sorted(
    set(
        [
            max(0, best_z - 2),
            max(0, best_z - 1),
            best_z,
            min(img.shape[2] - 1, best_z + 1),
            min(img.shape[2] - 1, best_z + 2),
        ]
    )
)

plt.figure(figsize=(18, 10))

for i, z in enumerate(candidate_slices):
    plt.subplot(len(candidate_slices), 3, 3 * i + 1)
    plt.imshow(img[:, :, z], cmap="gray")
    plt.title(f"Image z={z}")
    plt.axis("off")

    plt.subplot(len(candidate_slices), 3, 3 * i + 2)
    plt.imshow(gt[:, :, z])
    plt.title(f"GT z={z}")
    plt.axis("off")

    plt.subplot(len(candidate_slices), 3, 3 * i + 3)
    plt.imshow(pr[:, :, z])
    plt.title(f"Pred z={z}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("results/nearby_slices.png", dpi=150)
plt.show()


# ============================================================
# FINAL NOTE
# ============================================================

with open("results/example_results.md", "w") as f:
    f.write("# Example Results\n\n")
    f.write("The SwinUNETR model and pretrained backbone were loaded successfully.\n\n")
    f.write("Inference was performed without training, as required by the task.\n\n")
    f.write("Prediction quality is limited because the model was not fine-tuned on this spinal CT dataset.\n\n")
    f.write("Generated output files:\n")
    f.write("- multiclass_visualization.png\n")
    f.write("- binary_label_overlay.png\n")
    f.write("- nearby_slices.png\n")

print("\nFinal note:")
print("The SwinUNETR model and pretrained backbone were loaded successfully.")
print("Inference was performed without training, as required by the task.")
print("Prediction quality is limited because the model was not fine-tuned on this spinal CT dataset.")

torch.cuda.empty_cache()
print("\nDone.")
