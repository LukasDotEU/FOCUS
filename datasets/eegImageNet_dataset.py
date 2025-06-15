import os
import pandas as pd
import torch

from .base_dataset import BaseEEGDataset

class EEGImageNet(BaseEEGDataset):
    def __init__(self, eeg_root, images_root, use_images=False, images_file=None):
        """
        pth_file: path to the .pth containing:
          {
            'dataset': [  # list of samples
              {
                'eeg'    : Tensor [128, 500],
                'image'  : int (image index),
                'label'  : int (class index),
                'subject': int
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
        use_images: if False, __getitem__ returns no image
        """
        super().__init__(eeg_root=eeg_root, images_root=images_root, use_images=use_images, images_file=images_file)

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

        # 3) If use_images & preload_images, build an in-memory cache of all unique images
        if self.use_images:
            self.images = torch.load(os.path.join(self.images_root, images_file), weights_only=False)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns a dict:
          {
            'eeg'      : Tensor [ch (128), t (440)],
            'class_idx': int,
            'image_idx': int,
            'subject'  : int,
            'image'    : Tensor [feature_dimension] (if use_images=True, else not present)
          }
        """
        row = self.metadata.iloc[idx]  # Get the row for this index

        eeg = self.samples[idx]['eeg'][:,20:460] # Tensor [ch (128), 440]
        class_idx = row['class_idx']   # int
        image_idx = row['image_idx']   # int
        subject = row['subject']       # int

        if self.use_images:
            img_feature = self.images[self.class_labels[class_idx]][self.image_labels[image_idx]]
            return {
                'eeg': eeg,
                'class_idx': class_idx,
                'image_idx': image_idx,
                'subject': subject,
                'image': torch.from_numpy(img_feature)
            }
        else:
            return {
                'eeg': eeg,
                'class_idx': class_idx,
                'image_idx': image_idx,
                'subject': subject
            }
