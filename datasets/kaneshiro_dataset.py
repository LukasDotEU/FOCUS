import os
import pandas as pd
import scipy
import torch

from .base_dataset import BaseEEGDataset

# Mapping from class ID to human-readable class label
CLASS_LABELS = {
    1: 'Human Body',
    2: 'Human Face',
    3: 'Animal Body',
    4: 'Animal Face',
    5: 'Fruit Vegetable',
    6: 'Inanimate Object'
}

class Kaneshiro(BaseEEGDataset):
    def __init__(self, eeg_root, images_root, use_images=False, images_file=None, use_original=False):
        """
        eeg_root: directory containing .mat files
        Each S[1...10].mat contains keys:
          - 'exemplarLabels': numpy array shape (1, N) of image IDs
          - 'categoryLabels': numpy array shape (1, N) of class IDs (1-6)
          - 'sub'           : subject identifier string, e.g. 'S1'
          - 'X_3D'          : EEG data array shape (124, 32, N)
        Each S[1...10]_[a, b][1...3].mat contains keys:
          - 'exemplarLabels': numpy array shape (N, 1) of image IDs
          - 'categoryLabels': numpy array shape (N, 1) of class IDs (1-6)
          - 'sessionID'     : subject identifier string, e.g. 'S1' together with session identifier string, e.g. a1 concatenated with "_". 
          - 'xEpoched'      : EEG data array shape (129, 651, N)
        

        This loader reads all .mat files in the directory and flattens into a list of samples. Only the first 124 channels are being used.
        Sample dict keys:
          - 'eeg'    : Tensor [124, 651] or Tensor [124, 32]
          - 'image'  : int (image index)
          - 'label'  : int (class index)
          - 'subject': int (subject index)
        """
        super().__init__(eeg_root=eeg_root, images_root=images_root,
                         use_images=use_images, images_file=images_file)
        
        self.use_original = use_original

        self.samples = []
        self.class_labels = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS)] # altough not needed - 1-index based

        # List all .mat files
        files = os.listdir(eeg_root)
        mat_paths = sorted(
            os.path.join(eeg_root, f)
            for f in files
            if f.lower().endswith('.mat')
            and (
                (not self.use_original and "_" in f) 
                or (self.use_original and not "_" in f)
            )
        )

        records = []
        for mat_path in mat_paths:
            # Load .mat file
            mat = scipy.io.loadmat(mat_path)

            # Determine subject ID from 'sub' field in mat
            if self.use_original:
                sub_field:str = mat['sub'].item()
            else:
                sub_field:str = mat['sessionID'].item().split("_", 1)[0]
            # Parse integer after 'S'
            subject = int(sub_field.lstrip('S'))
            images_idxs = mat['exemplarLabels'].squeeze()  # shape (N,)
            class_idxs = mat['categoryLabels'].squeeze()   # shape (N,)
            if self.use_original:
                eeg_data = mat['X_3D']                     # shape (channels, time_steps, N)
            else:
                eeg_data = mat['xEpoched']                 # shape (channels, time_steps, N)
                eeg_data = eeg_data[:124, :, :]   # discard the last 4 channels + reference channel (129) which is zero-valued

            # Iterate trials
            for i in range(eeg_data.shape[-1]):
                eeg_trial = torch.from_numpy(eeg_data[:, :, i]).float()
                image_idx = images_idxs[i].item()
                class_idx = class_idxs[i].item()
                # Convert 1-based category and image to 0-based index
                image_idx = image_idx - 1
                class_idx = class_idx - 1

                self.samples.append(eeg_trial)
                
                records.append({
                	'idx': len(records),
                	'subject': subject,
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
            'eeg'      : Tensor [ch (124), t (651)] or Tensor [ch (124), t (32)],
            'class_idx': int,
            'image_idx': int,
            'subject'  : int,
            'image'    : Tensor [feature_dimension] (if use_images=True, else not present)
          }
        """
        row = self.metadata.iloc[idx]  # Get the row for this index

        eeg = self.samples[idx]        # Tensor [124, 651] or Tensor [124, 32]
        class_idx = row['class_idx']   # int
        image_idx = row['image_idx']   # int
        subject = row['subject']       # int

        if self.use_images:
            img_feature = self.images[class_idx + 1][image_idx + 1]
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
