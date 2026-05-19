"""
Extract CLIP features for images in dataset directories specified in config.py,
saving the features and corresponding folder names.
"""

import os


import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel


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


def process_dataset(images_root, subfolders, class_text_labels):
    """
    ClassLabels need to be in correct order.
    """
    # Setup device and model
    print("Initializing CLIP model and processor...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=".cache"
    )
    model = torch.nn.DataParallel(model).to(device)
    processor = CLIPProcessor.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=".cache"
    )

    image_paths = []  # full Paths
    class_ids = []

    # for individual image feature saving
    folder_names = []
    image_names = []
    text_prompts = []

    for condition_id, (subfolder, class_text_label) in enumerate(zip(subfolders, class_text_labels)):
        folder_path = os.path.join(images_root, subfolder)
        file_names = [
            e.name
            for e in os.scandir(folder_path)
            if e.is_file() and e.name.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        for fname in file_names:
            text_prompts.append(f"This picture is {class_text_label}")
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

    # === Extract and save individual image features ===
    all_feats = []

    for i, imgs in enumerate(loader):
        print(
            f"[ATMS Clip Features] Processing batch {i+1}/{len(loader)} ({len(imgs)} images)"
        )
        batch = processor(images=imgs, return_tensors="pt").pixel_values.to(
            device, non_blocking=True
        )
        with torch.no_grad():
            feats = model.module.get_image_features(batch)
        all_feats.append(feats.cpu())

    feats_tensor = torch.cat(all_feats, dim=0)
    feats_tensor = torch.nn.functional.normalize(feats_tensor, p=2, dim=1)

    # === Compute CLIP text embeddings for class labels ===
    text_inputs = processor(
        text=text_prompts, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    with torch.no_grad():
        label_feats = model.module.get_text_features(**text_inputs).cpu().numpy()

    norms = np.linalg.norm(label_feats, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)   # avoid div-by-zero
    label_feats = label_feats / norms


    # build dict {class_folder: {imagename: feat}}
    feature_dict = {}
    for feat_vec, folder_name, img_name, label_feat in zip(
        feats_tensor.numpy(), folder_names, image_names, label_feats
    ):
        feature_dict.setdefault(folder_name, {})[img_name] = np.stack([feat_vec, label_feat], axis=0)
    imgs_feat_path = os.path.join(images_root, "ATMS_clip_individual_features.pth")
    torch.save(feature_dict, imgs_feat_path)
    print(f"Saved individual ATMS Clip features at {imgs_feat_path}")

    # === Compute CLIP text embeddings for class labels ===
    print(f"Encoding {len(class_text_labels)} class text labels in CLIP space...")
    text_prompts = [f"This picture is {name}" for name in class_text_labels]
    text_inputs = processor(
        text=text_prompts, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    with torch.no_grad():
        label_feats = model.module.get_text_features(**text_inputs).cpu().numpy()

    norms = np.linalg.norm(label_feats, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)   # avoid div-by-zero
    label_feats = label_feats / norms

    result = {"clip_label_features": label_feats, "clip_label_prompts": text_prompts}

    labels_feat_path = os.path.join(images_root, "ATMS_clip_label_features.npy")
    # Save the result as a dataset-specific file
    np.save(labels_feat_path, result)
    print(f"Saved label features at {labels_feat_path}")
