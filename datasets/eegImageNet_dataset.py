import os
import argparse
import pandas as pd
import torch

import sys
# ensure local package import works when running as script
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datasets.base_dataset import BaseEEGDataset

PTH_FILE_DIR = {
    'eeg_55_95_std.pth': 'eegimagenet_individual_pt_55_95',
    'eeg_5_95_std.pth': 'eegimagenet_individual_pt_5_95',
    #'eeg_14_70_std.pth': 'eegimagenet_individual_pt_14_70'
}

def preprocess(pth_file: str, eeg_root: str):
    """
    Reads the .pth at eeg_root/pth_file, splits each sample into its own .pt file under:
      <eeg_root>/eegimagenet_individual_pt_.../
    Also generates:
      - metadata.csv with fields: idx, filepath, subject, class_idx, image_idx
      - class_labels.pt: list of class label strings
      - image_labels.pt: list of image label strings
    """
    out_dir = os.path.join(eeg_root, PTH_FILE_DIR[pth_file])
    os.makedirs(out_dir, exist_ok=True)

    data = torch.load(os.path.join(eeg_root, pth_file))
    # data['dataset'] is a list of dicts: each dict has keys 'eeg', 'image', 'label', 'subject'
    samples = data['dataset']
    class_labels = data.get('labels', [])
    image_labels = data.get('images', [])

    # save labels for use in Dataset
    torch.save(class_labels, os.path.join(out_dir, 'class_labels.pt'))
    torch.save(image_labels, os.path.join(out_dir, 'image_labels.pt'))

    # Build metadata DataFrame = one row per sample
    records = []
    for idx, sample in enumerate(samples): # idx is the index in the dataset
        eeg = sample['eeg']
        # trim time window
        eeg = eeg[:, 20:460]
        subj = sample['subject']
        class_idx = sample['label']
        image_idx = sample['image']

        fname = f"trial_{idx:05d}.pt"
        torch.save(eeg, os.path.join(out_dir, fname))

        records.append({
            'idx': idx,
            'filepath': fname,
            'subject': subj,
            'class_idx': class_idx,
            'image_idx': image_idx,
        })
        if idx % 50 == 0:
            print(f"processing idx {idx}")

    # write metadata
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, 'metadata.csv'), index=False)
    print(f"Saved {len(samples)} trials and labels to {out_dir}")

class EEGImageNet(BaseEEGDataset):
    def __init__(self, eeg_root: str, pth_file: str, images_root: str, 
                 use_images: bool = False, images_file: bool = None, use_cwt: bool = False):
        """
        PyTorch Dataset for preprocessed EEGImageNet samples.

        Expects directory:
           eeg_root/eegimagenet_individual_pt/
             - metadata.csv
             - trial_00000.pt, trial_00001.pt, ...
             - class_labels.pt
             - image_labels.pt
        use_images: if False, __getitem__ returns no image
        """
        super().__init__(eeg_root=eeg_root, images_root=images_root, 
                         use_images=use_images, images_file=images_file, use_cwt=use_cwt)
        self.pth_file = pth_file

        self.samples_dir = os.path.join(self.eeg_root, PTH_FILE_DIR[self.pth_file])
        self.metadata = pd.read_csv(os.path.join(self.samples_dir, 'metadata.csv'))

        # load saved labels
        self.class_labels = torch.load(os.path.join(self.samples_dir, 'class_labels.pt'))
        self.image_labels = torch.load(os.path.join(self.samples_dir, 'image_labels.pt'))

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
            'eeg'      : Tensor [ch (128), t (440)],
            'eeg'      : Tensor (if use_cwt=True, else not present),
            'class_idx': int,
            'image_idx': int,
            'subject'  : int,
            'image'    : Tensor [feature_dimension] (if use_images=True, else not present)
          }
        """
        row = self.metadata.iloc[idx]

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
            # lookup using loaded label lists
            class_label = self.class_labels[sample['class_idx']]
            img_label = self.image_labels[sample['image_idx']]
            feat = self.images[class_label][img_label]
            sample['image'] = torch.from_numpy(feat)
        return sample

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess EEGImageNet .pth to per-trial .pt')
    parser.add_argument('eeg_root', nargs='?', default='../Datasets/EEGImageNet/', help='Output root for individual .pt files')
    args = parser.parse_args()
    for pth_file in PTH_FILE_DIR.keys():
        preprocess(pth_file, args.eeg_root)