"""
Extract CLIP features for images in dataset directories specified in config.py,
saving the features and corresponding folder names.
"""

import os


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
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return img


# custom collate: batch of PIL images → list_of_PILs
def collate_fn(batch):
    return list(batch)


def process_dataset(images_root, subfolders):
    # Setup device and model
    print("Initializing CLIP model and processor...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14", cache_dir=".cache"
    )
    model = torch.nn.DataParallel(model).to(device)
    processor = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-large-patch14", cache_dir=".cache"
    )

    image_paths = []  # full Paths
    class_ids = []

    # for individual image feature saving
    folder_names = []
    image_names = []

    for condition_id, subfolder in enumerate(subfolders):
        folder_path = os.path.join(images_root, subfolder)
        file_names = [
            e.name
            for e in os.scandir(folder_path)
            if e.is_file() and e.name.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        for fname in file_names:
            image_paths.append(os.path.join(folder_path, fname))
            class_ids.append(condition_id)

            # for individual image feature saving
            folder_names.append(subfolder)
            image_names.append(fname.split(".", 1)[0])

    print(f"Found {len(image_paths)} images across {len(subfolders)} classes.")

    # Create global DataLoader
    dataset = GlobalImageDataset(image_paths)
    loader = DataLoader(
        dataset, batch_size=64, num_workers=4, pin_memory=True, collate_fn=collate_fn
    )

    all_feats = []  # list of CPU tensors

    for i, imgs in enumerate(loader):
        print(
            f"[NiceEEG Clip Features] Processing batch {i+1}/{len(loader)} ({len(imgs)} images)"
        )
        batch = processor(images=imgs, return_tensors="pt").pixel_values.to(
            device, non_blocking=True
        )
        with torch.no_grad():
            feats = model.module.get_image_features(batch)
        all_feats.append(feats.cpu())

    feats_tensor = torch.cat(all_feats, dim=0)
    # build dict {class_folder: {imagename: feat}}
    feature_dict = {}
    for feat_vec, folder_name, img_name in zip(
        feats_tensor.numpy(), folder_names, image_names
    ):
        feature_dict.setdefault(folder_name, {})[img_name] = feat_vec
    imgs_feat_path = os.path.join(images_root, "NICE_clip_individual_features.pth")
    torch.save(feature_dict, imgs_feat_path)
    print(f"Saved individual NiceEEG Clip features at {imgs_feat_path}")

    # Compute center features per folder
    dim = feats_tensor.size(1)
    num_classes = len(subfolders)
    ids_tensor = torch.tensor(class_ids)
    center_feats = torch.zeros((num_classes, dim), dtype=torch.float32)
    for fid in range(num_classes):
        mask = ids_tensor == fid
        if mask.sum() == 0:
            continue
        center_feats[fid] = feats_tensor[mask].mean(dim=0)

    result = {
        "clip_center_features": center_feats.numpy(),
        "clip_center_features_names": subfolders,
    }

    img_center_path = os.path.join(images_root, "NICE_clip_center_features.npy")
    np.save(img_center_path, result)
    print(f"Saved NiceEEG Clip features at {img_center_path}")
