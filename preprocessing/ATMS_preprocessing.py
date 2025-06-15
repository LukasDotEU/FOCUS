"""
Extract CLIP features for images in dataset directories specified in config.py,
saving the features and corresponding folder names.

WARNING: This script must be executed from the project directory
and not from within the preprocessing folder, as the relative paths will not work correctly.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to sys.path so config.py can be imported.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Dirty hack but okay for now..

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPImageProcessor, CLIPModel
from config import DATASET_CONFIGS

class GlobalImageDataset(Dataset):
    """
    Dataset over all images, returning (PIL_image, folder_id).
    """
    def __init__(self, image_paths, folder_ids):
        self.image_paths = image_paths
        self.folder_ids = folder_ids

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return img, self.folder_ids[idx]

def process_dataset(ds, model, processor, device, override=False):

    dataset_name = ds["name"]
    project_dir = Path(ds["images_root"])
    if dataset_name.startswith("ThingsEEG2"):
        # ThingsEEG2 has a different structure, so we need to adjust the path
        project_dir = project_dir / 'image_set'
    print(f"Processing dataset '{dataset_name}' at {project_dir}")

    output_path = project_dir / 'clip_center_features_ATMS-tests.npy'
    # Check if the output file already exists to avoid reprocessing
    if output_path.exists() and not override:
        print(f"Found existing output for '{dataset_name}', skipping.")
        return

    # Gather all image folder paths and corresponding folder IDs
    subfolders = sorted([entry.name for entry in os.scandir(project_dir) if entry.is_dir()])
    if dataset_name.startswith("ThingsEEG2"):
        # ThingsEEG2 has original test images in the same folder.
        # We want to repurpose the training images only.
        train_concepts = np.load(Path(ds["images_root"]) / 'image_metadata.npy', 
                                 allow_pickle=True).item()['train_img_concepts']
        train_concepts = [concept.split('_', 1)[-1] for concept in train_concepts]
        subfolders = [folder for folder in subfolders if folder in train_concepts]
        
    image_paths = []  # full Paths
    folder_ids   = []

    for condition_id, folder in enumerate(subfolders):
        folder_path = project_dir / folder
        file_names = [entry.name for entry in os.scandir(folder_path) 
                      if entry.is_file() and entry.name.lower().endswith(('.jpg', '.jpeg'))]
        for fname in file_names:
            image_paths.append(str(folder_path / fname))
            folder_ids.append(condition_id)

    total_imgs = len(image_paths)
    num_classes = len(subfolders)
    print(f"Found {total_imgs} images across {num_classes} folders.")

    # custom collate: batch of (PIL images, ids) → (list_of_PILs, tensor(ids))
    def collate_fn(batch):
        imgs, ids = zip(*batch)
        return list(imgs), torch.tensor(ids, dtype=torch.long)

    # Create global DataLoader
    dataset = GlobalImageDataset(image_paths, folder_ids)
    loader  = DataLoader(dataset,
                         batch_size=64,
                         num_workers=4,
                         pin_memory=True,
                         collate_fn=collate_fn)
    
    # Prepare storage for features and folder_ids
    all_feats = []  # list of CPU tensors
    all_ids   = []  # list of lists

    total_batches = len(loader)
    # Single pass over all images
    for i, (imgs, ids) in enumerate(loader):
        print(f"[{dataset_name}] Processing batch {i+1}/{total_batches} ({len(imgs)} images)")
        batch = processor(images=imgs, return_tensors='pt').pixel_values.to(device, non_blocking=True)
        with torch.no_grad():
            feats = model.module.get_image_features(batch)
        all_feats.append(feats.cpu())
        all_ids.append(ids)
    
    feats_tensor = torch.cat(all_feats, dim=0)  # [total_imgs, dim]
    ids_tensor   = torch.cat(all_ids, dim=0)   # [total_imgs]

    # Compute center features per folder
    dim = feats_tensor.size(1)
    center_feats = torch.zeros((num_classes, dim), dtype=torch.float32)
    for fid in range(num_classes):
        mask = ids_tensor == fid
        if mask.sum() == 0:
            continue
        center_feats[fid] = feats_tensor[mask].mean(dim=0)
    
    result = {
        'clip_center_features': center_feats.numpy(),
        'clip_center_features_names': subfolders
    }

    # Special reorder for EEGImageNet as original class_id order is not alphabetical.
    if "EEGImageNet" == dataset_name:
        print("Reordering EEGImageNet classes...")
        # Load the EEG file that contains the class names (labels)
        eeg_path = Path(ds["eeg_root"])
        eeg_data = torch.load(eeg_path)
        target_order = eeg_data["labels"]

        # Reorder the features and names based on the target order.
        # Assumes that the names in target_order match those in subfolders.
        orig = result['clip_center_features']
        reordered = np.stack([orig[subfolders.index(lbl)] for lbl in target_order])
        # Overwrite the dictionary with reordered entries.
        result = {
            "clip_center_features": reordered,
            "clip_center_features_names": target_order,
        }

    # Save the result as a dataset-specific file
    np.save(output_path, result)
    print(f"Saved features for dataset '{dataset_name}' at {output_path}")

def main():
    # Setup device and model
    print("Initializing CLIP model and processor...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLIPModel.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K', cache_dir='.cache')
    model = torch.nn.DataParallel(model).to(device)
    processor = CLIPImageProcessor.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K', cache_dir='.cache')

    processed = []
    for ds in DATASET_CONFIGS:
        # Check if the project directory has already been processed
        if ds["images_root"] in processed:
            print(f"Dataset '{ds['name']}' (or similar) already processed, skipping.")
            continue
        processed.append(ds["images_root"])

        process_dataset(ds, model, processor, device, override=False)


if __name__ == '__main__':
    main()
