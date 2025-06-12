import os
import numpy as np
import pandas as pd
import torch
from PIL import Image

from .base_dataset import BaseEEGDataset

class ThingsEEG2(BaseEEGDataset):
    def __init__(self,
                 eeg_root: str,
                 images_root: str,
                 use_images: bool = False,
                 preload_images: bool = False,
                 image_transform=None,
                 average_reps: bool = False):
        """
        Adaptation for the ThingsEEG2 layout:
        - eeg_root/
            sub-01/
               preprocessed_eeg_training.npy  # dict with key "preprocessed_eeg_data"
            sub-02/
               ...
        - images_root/
            image_metadata.npy  # dict with keys "train_img_files", "train_img_concepts"
            image_set/
                aardvark/
                    aardvark_01b.jpg
                    ...
                air_conditioner/
                    air_conditioner_02a.jpg
                ...
        """
        super().__init__(eeg_root=eeg_root,
                         images_root=images_root,
                         use_images=use_images,
                         preload_images=preload_images,
                         image_transform=image_transform)
        
        self.average_reps = average_reps  # If True, average the 4 repetitions of each image

        # --- 1) Load and concatenate EEG data from all subjects --------------------
        self.samples = []
        for subfolder in sorted(os.listdir(self.eeg_root)):
            if not subfolder.startswith("sub-"):
                continue
            subject_id = subfolder.split("-")[1]
            npy_path = os.path.join(self.eeg_root, subfolder, "preprocessed_eeg_training.npy")
            data_dict = np.load(npy_path, allow_pickle=True)
            eeg_array:np.ndarray = data_dict["preprocessed_eeg_data"]
            # eeg_array shape: (n_images=16540, n_reps=4, n_chans=63, n_timesteps=250)
            n_images, n_reps, _, _ = eeg_array.shape

            # flatten out the reps so each trial is its own sample
            if self.average_reps:
                # Average the 4 repetitions for each image
                eeg_array = eeg_array.mean(axis=1)
                for img_idx in range(n_images):
                    self.samples.append({
                        "eeg": torch.from_numpy(eeg_array[img_idx]).to(torch.float32),  # shape [63,250]
                        "image_idx": img_idx,
                        "class_idx": img_idx // 10,  # assuming 10 images per class
                        "subject": int(subject_id),
                    })
            else:
                for img_idx in range(n_images):
                    for rep in range(n_reps):
                        self.samples.append({
                            "eeg": torch.from_numpy(eeg_array[img_idx, rep]).to(torch.float32),  # shape [63,250]
                            "image_idx": img_idx,
                            "class_idx": img_idx // 10,  # assuming 10 images per class
                            "subject": int(subject_id),
                        })

        # --- 2) Load image metadata -----------------------------------------------
        meta = np.load(os.path.join(self.images_root, "image_metadata.npy"),
                       allow_pickle=True).item()
        self.image_labels:list[str] = meta["train_img_files"]   # e.g. ["aardvark_01b.jpg", ...]
        concept_labels:list[str] = meta["train_img_concepts"]   # e.g. ["00001_aardvark", ...]
        # Build a mapping from image_idx → class_idx, class_label:
        self.class_labels = []  # unique list of labels, e.g. ["aardvark", "air_conditioner", ...]
        for idx, concept in enumerate(concept_labels):
            if idx % 10 == 0:
                class_label = concept.split("_", 1)[1]
                self.class_labels.append(class_label)
        self.image_labels = [label.rsplit(".", 1)[0] for label in self.image_labels]  # remove file extensions

        # --- 3) Build metadata DataFrame ------------------------------------------
        records = []
        for idx, sample in enumerate(self.samples):
            records.append({
                "idx": idx,
                "subject": sample["subject"],
                "image_idx": sample["image_idx"],
                "class_idx": sample["class_idx"],
            })
        self.metadata = pd.DataFrame(records)

        # --- 4) (Optional) preload images into RAM -------------------------------
        self._image_cache = {}
        if self.use_images and self.preload_images:
            for idx, class_idx, image_idx in zip(self.metadata['idx'], self.metadata['class_idx'], self.metadata['image_idx']):
                self._image_cache[idx] = self._load_image(class_idx, image_idx)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns a dict:
          {
            'eeg'      : Tensor [ch (63), t],
            'class_idx': int,
            'image_idx': int,
            'subject'  : int,
            'image'    : Tensor [3, X (224), Y (224)] (if use_images=True, else not present)
          }
        """
        row = self.metadata.iloc[idx]  # Get the row for this index
        
        eeg = self.samples[idx]['eeg'] # Tensor [ch (63), t]
        class_idx = row['class_idx']   # int
        image_idx = row['image_idx']   # int
        subject = row['subject']       # int

        if self.use_images:
            if self.preload_images:
                img_tensor = self._image_cache[idx]
            else:
                img_tensor = self._load_image(class_idx, image_idx)
            return {
                'eeg': eeg,
                'class_idx': class_idx,
                'image_idx': image_idx,
                'subject': subject,
                'image': img_tensor
            }
        
        return {
            'eeg': eeg,
            'class_idx': class_idx,
            'image_idx': image_idx,
            'subject': subject
        }

    def _load_image(self, class_idx, image_idx):
        """
        We load images_root/image_set/class_label/{image_label}.jpg, apply transforms, and return a torch.Tensor.
        """
        filename = f"{self.image_labels[image_idx]}.jpg"
        img_path = os.path.join(self.images_root, "image_set",self.class_labels[class_idx], filename)

        with Image.open(img_path).convert('RGB') as img:
            # processor returns a batch; grab the single sample
            processed = self.processor(images=img, return_tensors="pt")
            # pixel_values has shape (1, 3, H, W) → drop batch dim
            pixel_values = processed.pixel_values.squeeze(0)
            return pixel_values
