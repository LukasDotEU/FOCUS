import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from .base_dataset import BaseEEGDataset

class EEGImageNet(BaseEEGDataset):
    def __init__(self, eeg_root, images_root,
                 use_images=False, preload_images=False, use_image_transform=False,
                 image_transform=None):
        """
        pth_file: path to the .pth containing:
          {
            'dataset': [  # list of samples
              {
                'eeg'    : Tensor [128, 500],
                'image'  : int (image index),
                'label'  : int (class index),
                'subject': any (e.g. int)
              },
              ...
            ],
            'labels': [class_label_0, class_label_1, ...],
            'images': [image_label_0, image_label_1, ...]
                     # where each image_label is a string like "cat_0123"
          }
        images_root: root directory for the image files, structured as:
          images_root/
            class_label_0/
              class_label_0_imageLabel0.JPEG
              ...
            class_label_1/
              ...
        use_images: if False, __getitem__ returns {'image': None}
        preload_images: if True, load all images into memory at init
        image_transform: torchvision transforms to apply to PIL images
        """
        super().__init__(eeg_root=eeg_root, images_root=images_root, use_images=use_images, preload_images=preload_images)

        # 1) Load the .pth file
        data:dict = torch.load(self.eeg_root)
        # data['dataset'] is a list of dicts: each dict has keys 'eeg', 'image', 'label', 'subject'
        self.samples = data['dataset']
        self.class_labels = data.get('labels', [])
        self.image_labels = data.get('images', [])

        # 2) Build metadata DataFrame = one row per sample
        records = []
        for idx, sample in enumerate(self.samples): # idx is the index in the dataset
            subj = sample['subject']
            class_idx = sample['label']
            image_idx = sample['image']
            records.append({
                'idx': idx,
                'subject': subj,
                'class_idx': class_idx,
                'image_idx': image_idx,
            })
        self.metadata = pd.DataFrame(records)

        # 3) Set up image transform (if none provided, use a default CLIP‐style pipeline)
        if use_image_transform and image_transform is None:
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        elif use_image_transform:
            self.image_transform = image_transform
        else:
            self.image_transform = None

        # 4) If use_images & preload_images, build an in-memory cache of all unique images
        self._image_cache = {}
        if self.use_images and self.preload_images:
            for idx, class_label, image_label in zip(self.metadata['idx'], self.class_labels[self.metadata['class_idx']], self.image_labels[self.metadata['image_idx']]):
                self._image_cache[idx] = self._load_image(class_label, image_label)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns a dict:
          {
            'eeg'      : Tensor [128, 500],
            'class_idx': int,
            'image_idx': int,
            'subject'  : int,
            'image'    : Tensor [3, X, Y] or None
          }
        """
        row = self.metadata[self.metadata['idx'] == idx].iloc[0]  # Get the row for this index

        eeg = self.samples[idx]['eeg'][:,20:460] # Tensor [440, 128]
        class_idx = row['class_idx']   # int
        image_idx = row['image_idx']   # int
        subject = row['subject']       # int
        img_tensor = None

        if self.use_images:
            if self.preload_images:
                img_tensor = self._image_cache[idx]
            else:
                img_tensor = self._load_image(self.class_labels[class_idx], self.image_labels[image_idx])
            return {
                'eeg': eeg,
                'class_idx': class_idx,
                'image_idx': image_idx,
                'subject': subject,
                'image': img_tensor
            }
        else:
            return {
                'eeg': eeg,
                'class_idx': class_idx,
                'image_idx': image_idx,
                'subject': subject
            }

    def _load_image(self, class_label, image_label):
        """
        Given an image index, look up its label string in self.image_labels.
        Each label is formatted as '{class_label}_{image_label}'. We split
        on the first underscore to find the class subfolder, then load
        images_root/class_label/{class_label}_{image_label}.JPEG, apply
        transforms, and return a torch.Tensor.
        """
        filename = f"{image_label}.JPEG"
        img_path = os.path.join(self.images_root, class_label, filename)

        with Image.open(img_path).convert('RGB') as img:
            if self.image_transform is None:
                return img
            else:
                return self.image_transform(img)
