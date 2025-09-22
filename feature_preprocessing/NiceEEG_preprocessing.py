"""
Extract CLIP features for images in dataset directories specified in config.py,
saving the features and corresponding folder names.
"""

import os
from pathlib import Path


import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPImageProcessor, CLIPModel

class GlobalImageDataset(Dataset):
    """
    Dataset over all images, returning PIL_image.
    """
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return img

def process_dataset(images_root, images_file_path, eeg_file = None, ThingsEEG2 = False):
    # Setup device and model
    print("Initializing CLIP model and processor...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14', cache_dir='.cache')
    model = torch.nn.DataParallel(model).to(device)
    processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14', cache_dir='.cache')

    project_dir = Path(images_root)
    if ThingsEEG2:  # ThingsEEG2 has a different structure, so we need to adjust the path
        concepts_root = images_root
        project_dir = project_dir / 'image_set'
    print(f"Processing NiceEEG Clip features at {project_dir}")

    # Gather all image folder paths and corresponding folder IDs
    subfolders = sorted([entry.name for entry in os.scandir(project_dir) if entry.is_dir()])
    if ThingsEEG2:  # ThingsEEG2 has original test images in the same folder. We want to repurpose the training images only.
        train_concepts = np.load(concepts_root / 'image_metadata.npy', 
                                 allow_pickle=True).item()['train_img_concepts']
        train_concepts = [concept.split('_', 1)[-1] for concept in train_concepts]
        subfolders = [folder for folder in subfolders if folder in train_concepts]
        
    image_paths = []  # full Paths
    class_ids  = []

    # for individual image feature saving
    class_labels = []
    image_labels = []

    for condition_id, folder in enumerate(subfolders):
        folder_path = project_dir / folder
        file_names = [entry.name for entry in os.scandir(folder_path) 
                      if entry.is_file() and entry.name.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for fname in file_names:
            image_paths.append(str(folder_path / fname))
            class_ids.append(condition_id)

            # for individual image feature saving
            class_labels.append(folder)
            image_labels.append(fname.split('.',1)[0])

    total_imgs = len(image_paths)
    num_classes = len(subfolders)
    print(f"Found {total_imgs} images across {num_classes} folders.")

    # custom collate: batch of PIL images → list_of_PILs
    def collate_fn(batch):
        return list(batch)

    # Create global DataLoader
    dataset = GlobalImageDataset(image_paths)
    loader  = DataLoader(dataset,
                         batch_size=64,
                         num_workers=4,
                         pin_memory=True,
                         collate_fn=collate_fn)
    
    # Prepare storage for features
    all_feats = []  # list of CPU tensors

    total_batches = len(loader)
    # Single pass over all images
    for i, (imgs) in enumerate(loader):
        print(f"[NiceEEG Clip Features] Processing batch {i+1}/{total_batches} ({len(imgs)} images)")
        batch = processor(images=imgs, return_tensors='pt').pixel_values.to(device, non_blocking=True)
        with torch.no_grad():
            feats = model.module.get_image_features(batch)
        all_feats.append(feats.cpu())
    
    feats_tensor = torch.cat(all_feats, dim=0)  # [total_imgs, dim]
    ids_tensor   = torch.tensor(class_ids)      # [total_imgs]

    # Build the nested dict:
    feature_dict = {}
    for feat_vec, class_label, img_label in zip(feats_tensor.numpy(), class_labels, image_labels):
        if class_label not in feature_dict:
            feature_dict[class_label] = {}
        feature_dict[class_label][img_label] = feat_vec
    torch.save(feature_dict, images_file_path)
    print(f"Saved individual NiceEEG Clip features at {images_file_path}")

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
    if eeg_file is not None:
        print("Reordering EEGImageNet classes...")
        # Load the EEG file that contains the class names (labels)
        eeg_data = torch.load(Path(eeg_file))
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
    
    images_center_file = project_dir / 'NICE_clip_center_features.npy'

    # Save the result as a dataset-specific file
    np.save(images_center_file, result)
    print(f"Saved NiceEEG Clip features at {images_center_file}")
