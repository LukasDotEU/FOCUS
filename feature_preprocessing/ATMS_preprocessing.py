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
from transformers import CLIPProcessor, CLIPModel
from config import DATASET_CONFIGS

# Mapping for Kaneshiro numeric labels\NAMESPACE
KANESHIRO_LABELS = {
    1: 'Human Body',
    2: 'Human Face',
    3: 'Animal Body',
    4: 'Animal Face',
    5: 'Fruit Vegetable',
    6: 'Inanimate Object'
}

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


def process_dataset(ds, model, processor, device, override=False):

    dataset_name = ds["name"]
    project_dir = Path(ds["images_root"])
    if dataset_name.startswith("ThingsEEG2"):
        # ThingsEEG2 has a different structure, so we need to adjust the path
        project_dir = project_dir / 'image_set'
    print(f"Processing dataset '{dataset_name}' at {project_dir}")

    output_path = project_dir / 'ATMS_clip_label_features.npy'
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
    class_ids   = []

    # for individual image feature saving
    class_labels = []
    image_labels = []

    for condition_id, class_label in enumerate(subfolders):
        folder_path = project_dir / class_label
        files = [e.name for e in os.scandir(folder_path) if e.is_file() and e.name.lower().endswith(('.jpg','.png','.jpeg'))]
        for fname in files:
            image_paths.append(str(folder_path / fname))
            class_ids.append(condition_id)

            # for individual image feature saving
            class_labels.append(class_label)
            image_labels.append(Path(fname).stem)

    print(f"Found {len(image_paths)} images across {len(subfolders)} classes.")

    # custom collate: batch of PIL images → list_of_PILs
    def collate_fn(batch):
        return list(batch)

    # Create global DataLoader
    dataset = GlobalImageDataset(image_paths)
    loader  = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True, collate_fn=collate_fn)

    # === Extract and save individual image features ===
    all_feats = []
    for i, imgs in enumerate(loader):
        print(f"[{dataset_name}] Batch {i+1}/{len(loader)}")
        batch = processor(images=imgs, return_tensors='pt').pixel_values.to(device, non_blocking=True)
        with torch.no_grad():
            feats = model.module.get_image_features(batch)
        all_feats.append(feats.cpu())

    feats_tensor = torch.cat(all_feats, dim=0)
    # build dict {class_folder: {imagename:feat}}
    feature_dict = {}
    for feat_vec, class_label, img_label in zip(feats_tensor.numpy(), class_labels, image_labels):
        feature_dict.setdefault(class_label, {})[img_label] = feat_vec
    torch.save(feature_dict, project_dir / "ATMS_clip_individual_features.pth")
    print(f"Saved individual image features at {project_dir / 'ATMS_clip_individual_features.pth'}")

    # === Compute CLIP text embeddings for class labels ===
    if dataset_name.startswith("Kaneshiro"):
        text_labels = [KANESHIRO_LABELS[int(f)] for f in subfolders]
    else:
        text_labels = subfolders

    print(f"Encoding {len(text_labels)} class text labels in CLIP space...")
    text_inputs = processor(text=text_labels, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        label_feats = model.module.get_text_features(**text_inputs).cpu().numpy()

    result = {
        'clip_label_features': label_feats,
        'clip_label_names': text_labels
    }

    if dataset_name == "EEGImageNet":
        print("Reordering EEGImageNet classes...")
        eeg_data = torch.load(os.path.join(ds["eeg_root"], 'eeg_55_95_std.pth'))
        order = eeg_data["labels"]
        orig_feats = result['clip_label_features']
        orig_names = result['clip_label_names']
        reordered = np.stack([orig_feats[orig_names.index(lbl)] for lbl in order])
        result = { 'clip_label_features': reordered, 'clip_label_names': order }

    # Save the result as a dataset-specific file
    np.save(output_path, result)
    print(f"Saved label features at {output_path}")


def main():
    # Setup device and model
    print("Initializing CLIP model and processor...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLIPModel.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K', cache_dir='.cache')
    model = torch.nn.DataParallel(model).to(device)
    processor = CLIPProcessor.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K', cache_dir='.cache')

    seen = set()
    for ds in DATASET_CONFIGS:
        # Check if the project directory has already been processed
        root = ds['images_root']
        if root in seen:
            print(f"Already processed {root} (or similar), skipping.")
            continue
        seen.add(root)
        process_dataset(ds, model, processor, device, override=False)

if __name__ == '__main__':
    main()
