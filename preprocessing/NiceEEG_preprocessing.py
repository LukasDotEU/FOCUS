"""
Extract CLIP features for images in dataset directories specified in config.py,
saving the features and corresponding folder names.

WARNING: This script must be executed from the project directory
and not from within the preprocessing folder, as the relative paths will not work correctly.
"""

import sys
import os

# Add the parent directory to sys.path so config.py can be imported.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Dirty hack but okay for now..

import torch
import numpy as np
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel
from config import DATASET_CONFIGS

# Initialize CLIP model and processor once (using multiple GPUs)
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=".cache")
model = torch.nn.DataParallel(model).cuda()
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=".cache")

processed = []
OVERRIDE = False  # Set to True to reprocess all datasets regardless of existing files

for ds in DATASET_CONFIGS:
    # Process only datasets with an image directory specified.
    if "images_root" not in ds:
        continue

    dataset_name = ds["name"]
    project_dir = ds["images_root"]
    print(f"Processing dataset '{dataset_name}' at {project_dir}")

    output_path = os.path.join(project_dir, f"clip_center_features.npy")
    # Check if the output file already exists to avoid reprocessing
    if os.path.exists(output_path) and not OVERRIDE:
        print(f"File for dataset '{dataset_name}' (or similar) already exists, skipping.")
        continue

    # Check if the project directory has already been processed
    if project_dir in processed:
        print(f"Dataset '{dataset_name}' (or similar) already processed, skipping.")
        continue
    processed.append(project_dir)

    # Get a sorted list of first-level subdirectories
    # Important as this ensures consistent ordering of conditions
    subfolders = sorted(
        [d for d in os.listdir(project_dir) if os.path.isdir(os.path.join(project_dir, d))]
    )

    # Preallocate tensor for center features based on number of subfolders
    center_features = torch.zeros((len(subfolders), 768), dtype=torch.float32).cuda()

    for condition_id, folder in enumerate(subfolders):
        folder_path = os.path.join(project_dir, folder)
        # List only image files in the folder
        files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg"))
        ]
        print(f"Processing condition {condition_id} ('{folder}') with {len(files)} images", flush=True)

        # Allocate tensor for features of images in this folder
        class_features = torch.zeros((len(files), 768), dtype=torch.float32)
        for file_id, file in enumerate(files):
            img_path = os.path.join(folder_path, file)
            img = Image.open(img_path).convert("RGB")
            processed = processor(images=img, return_tensors="pt")
            processed = processed.pixel_values.cuda()

            with torch.no_grad():
                x = model.module.get_image_features(processed)
            class_features[file_id] = torch.squeeze(x)

        # Compute the mean feature for the folder
        center_features[condition_id] = torch.mean(class_features, dim=0)

    # Build the output dictionary with folder names (current order)
    center_features_names = {
        "clip_center_features": center_features.detach().cpu().numpy(),
        "clip_center_features_names": subfolders,
    }

    # Reorder features for EEGImageNet as original class_id order is not alphabetical.
    if "EEGImageNet" == dataset_name:
        print("Reordering features based on EEG labels...")
        # Load the EEG file that contains the class names (labels)
        eeg_path = ds["eeg_root"]
        # Resolve relative path if needed:
        eeg_path = os.path.abspath(os.path.join(os.getcwd(), eeg_path))
        eeg_data = torch.load(eeg_path)
        target_order = eeg_data["labels"]

        # Reorder the features and names based on the target order.
        # Assumes that the names in target_order match those in subfolders.
        orig_features = center_features.detach().cpu().numpy()
        reordered_features = []
        for label in target_order:
            try:
                idx = subfolders.index(label)
                reordered_features.append(orig_features[idx])
            except ValueError:
                raise ValueError(f"Label '{label}' from EEG file not found in folder names: {subfolders}")
        # Convert list to numpy array
        reordered_features = np.stack(reordered_features)
        # Overwrite the dictionary with reordered entries.
        center_features_names = {
            "clip_center_features": reordered_features,
            "clip_center_features_names": target_order,
        }

    # Save the result as a dataset-specific file
    np.save(output_path, center_features_names)
    print(f"Saved features for dataset '{dataset_name}' at {output_path}")