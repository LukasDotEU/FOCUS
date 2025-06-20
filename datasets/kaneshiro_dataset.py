import os
import argparse
import scipy.io
import torch
import pandas as pd

import sys
# ensure local package import works when running as script
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datasets.base_dataset import BaseEEGDataset

# Mapping from class ID to human-readable class label
CLASS_LABELS = {
    1: 'Human Body',
    2: 'Human Face',
    3: 'Animal Body',
    4: 'Animal Face',
    5: 'Fruit Vegetable',
    6: 'Inanimate Object'
}

def preprocess(eeg_root: str, use_original: bool):
    """
    Splits multi-trial .mat files into individual .pt files.
    Creates a subfolder under eeg_root:
      - 'kaneshiro_individual_pt_original'  (use_original=True)
      - 'kaneshiro_individual_pt_updated'  (use_original=False)
    Also writes a metadata.csv in that folder.

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
    
    This function reads all .mat files in the directory and flattens into a list of samples. Only the first 124 channels are being used.
    Sample dict keys:
      - 'eeg'     : Tensor [124, 651] or Tensor [124, 32]
      - 'filename': str
      - 'image'   : int (image index)
      - 'label'   : int (class index)
      - 'subject' : int (subject index)
    """
    mode = 'original' if use_original else 'updated'
    out_dir = os.path.join(eeg_root, f'kaneshiro_individual_pt_{mode}')
    os.makedirs(out_dir, exist_ok=True)

    records = []
    global_idx = 0

    # select .mat files
    all_files = sorted(f for f in os.listdir(eeg_root) if f.lower().endswith('.mat'))
    def sel(f): return (use_original and '_' not in f) or (not use_original and '_' in f)
    mat_paths = [os.path.join(eeg_root, f) for f in all_files if sel(f)]

    for mat_path in mat_paths:
        mat = scipy.io.loadmat(mat_path)
        if use_original:
            sub_field = mat['sub'].item()
            eeg_data = mat['X_3D']          # (channels, time_steps, N)
        else:
            sub_field = mat['sessionID'].item().split('_', 1)[0]
            eeg_data = mat['xEpoched'][:124, :, :]  # discard the last 4 channels + reference channel (129) which is zero-valued; (channels, time_steps, N)

        subject = int(sub_field.lstrip('S'))
        images = mat['exemplarLabels'].squeeze()
        classes = mat['categoryLabels'].squeeze()

        for i in range(eeg_data.shape[-1]):
            trial = torch.from_numpy(eeg_data[:, :, i]).float()
            image_idx = int(images[i].item()) - 1
            class_idx = int(classes[i].item()) - 1

            fname = f"trial_{global_idx:05d}.pt"
            torch.save(trial, os.path.join(out_dir, fname))

            records.append({
                'idx': global_idx,
                'filepath': fname,
                'subject': subject,
                'class_idx': class_idx,
                'image_idx': image_idx
            })
            global_idx += 1
        print(f"Processing file {mat_path}")

    # write metadata
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, 'metadata.csv'), index=False)
    print(f"[{mode.upper()}] Saved {global_idx} trials to {out_dir}")


class Kaneshiro(BaseEEGDataset):
    """
    Dataset for preprocessed Kaneshiro EEG trials.

    Args:
        eeg_root: root directory containing 'kaneshiro_individual_pt_original' and/or
                  'kaneshiro_individual_pt_updated'.
        use_images: if True, load image features from images_root/images_file.
        use_original: if True, loads from the 'original' folder; else 'updated'.
    """
    def __init__(self, eeg_root: str, images_root: str, use_original: bool,
                 use_images: bool = False, images_file: str = None, use_cwt: bool = False):
        super().__init__(eeg_root=eeg_root, images_root=images_root,
                         use_images=use_images, images_file=images_file, use_cwt=use_cwt)
        self.use_original = use_original

        mode = 'original' if self.use_original else 'updated'
        samples_dir = os.path.join(self.eeg_root, f'kaneshiro_individual_pt_{mode}')
        meta_path = os.path.join(samples_dir, 'metadata.csv')
        self.metadata = pd.read_csv(meta_path)
        self.samples_dir = samples_dir

        self.eeg_cache = {}
        if self.use_cwt:
            self.cwt_cache = {}

        if self.use_images:
            self.images = torch.load(os.path.join(self.images_root, self.images_file), weights_only=False)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns a dict:
          {
            'eeg'      : Tensor [ch (124), t (651)] or Tensor [ch (124), t (32)],
            'eeg'      : Tensor (if use_cwt=True, else not present),
            'class_idx': int,
            'image_idx': int,
            'subject'  : int,
            'image'    : Tensor [feature_dimension] (if use_images=True, else not present)
          }
        """
        row = self.metadata.iloc[idx]  # Get the row for this index

        if idx not in self.eeg_cache:
            eeg_path = os.path.join(self.samples_dir, row['filepath'])
            self.eeg_cache[idx] = torch.load(eeg_path)

            if self.use_cwt:
                cwt_path = os.path.join(self.samples_dir, "cwt_" + row['filepath'])
                self.cwt_cache[idx] = torch.load(cwt_path)
        
        sample = {
            'eeg': self.eeg_cache[idx],
            'class_idx': int(row['class_idx']),
            'image_idx': int(row['image_idx']),
            'subject': int(row['subject'])
        }

        if self.use_cwt:
            sample['cwt'] = self.cwt_cache[idx]

        if self.use_images:
            feat = self.images[str(sample['class_idx'] + 1)][str(sample['image_idx'] + 1)]
            sample['image'] = torch.from_numpy(feat)
        return sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess Kaneshiro .mat files and/or test dataset loader'
    )
    parser.add_argument('eeg_root', nargs='?', default='../Datasets/Kaneshiro/', help='Root directory for .mat data')
    args = parser.parse_args()

    # preprocess updated and original data
    preprocess(args.eeg_root, use_original=False)
    preprocess(args.eeg_root, use_original=True)
