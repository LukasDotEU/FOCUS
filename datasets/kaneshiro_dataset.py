import os
import scipy.io
import torch
import pandas as pd
import concurrent.futures

from datasets.base_dataset import BaseEEGDataset
from feature_preprocessing import CAWMASASTST_eegcwt_preprocessing

# Mapping from class ID to human-readable class label
CLASS_TEXT_DICT = {
    '1': 'Human Body',
    '2': 'Human Face',
    '3': 'Animal Body',
    '4': 'Animal Face',
    '5': 'Fruit Vegetable',
    '6': 'Inanimate Object'
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
    out_dir = os.path.join(eeg_root, f'kaneshiro{mode}_individual_pt')
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
                'filename': fname,
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
    def __init__(self, eeg_root: str, images_root: str, sampling_rate: float, use_original: bool,
                 use_images: bool = False, images_file: str = None, use_cwt: bool = False, pre_load: bool = True):
        super().__init__(eeg_root=eeg_root, images_root=images_root, sampling_rate=sampling_rate,
                         use_images=use_images, images_file=images_file, use_cwt=use_cwt, pre_load=pre_load)
        self.use_original = use_original

        mode = 'original' if self.use_original else 'updated'
        samples_dir = os.path.join(self.eeg_root, f'kaneshiro{mode}_individual_pt')
        if not os.path.exists(samples_dir):
            print(f"Preprocessing not done before. Preprocesing {mode} data into individual .pt files...")
            preprocess(self.eeg_root, use_original=self.use_original)

        meta_path = os.path.join(samples_dir, 'metadata.csv')
        self.metadata = pd.read_csv(meta_path)
        self.metadata = (self.metadata.sort_values('idx').set_index('idx', drop=False))  # Ensure sorted by idx
        self.samples_dir = samples_dir

        # depracation compliance when filename column was named filepath
        if 'filepath' in self.metadata.columns:
            self.metadata.rename(columns={'filepath': 'filename'}, inplace=True)
        
        self._filenames = self.metadata['filename'].tolist()
        if self.use_cwt:
            self._cwt_names = ["cwt_" + fp for fp in self._filenames]
            if not all(os.path.exists(os.path.join(self.samples_dir, fname)) for fname in self._cwt_names):
                print("CWT files not found, computing CWT for all trials...")
                CAWMASASTST_eegcwt_preprocessing.process_dataset(self.samples_dir, self.sampling_rate)

        if self.pre_load:
            def load_pt(fname):
                return torch.load(os.path.join(self.samples_dir, fname))
            # Pre-load all EEG data
            with concurrent.futures.ThreadPoolExecutor() as executor:
                self.eeg_data = self.eeg_data = list(executor.map(load_pt, self._filenames))
            if self.use_cwt:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    self.cwt_data = list(executor.map(load_pt, self._cwt_names))

        if self.use_images:
            images_file_path = os.path.join(self.images_root, self.images_file)
            if not os.path.exists(images_file_path):
                subfolders = sorted([entry.name for entry in os.scandir(self.images_root) if entry.is_dir()])
                if self.images_file.startswith('ATMS'):
                    from feature_preprocessing.ATMS_preprocessing import process_dataset
                    class_text_labels = [CLASS_TEXT_DICT[label] for label in subfolders]
                    process_dataset(self.images_root, subfolders, class_text_labels)
                elif self.images_file.startswith('NICE'):
                    from feature_preprocessing.NiceEEG_preprocessing import process_dataset
                    process_dataset(self.images_root, subfolders)
                elif self.images_file.startswith('EEGClip'):
                    from feature_preprocessing.EEGClip_preprocessing import process_dataset
                    process_dataset(self.images_root, subfolders)
            
            self.images = torch.load(images_file_path, weights_only=False)

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

        if self.pre_load:
            eeg = self.eeg_data[idx]
            if self.use_cwt:
                cwt = self.cwt_data[idx]
        else:
            eeg = torch.load(os.path.join(self.samples_dir, row['filename']))
            if self.use_cwt:
                cwt = torch.load(os.path.join(self.samples_dir, "cwt_" + row['filename']))
        
        sample = {
            'eeg': eeg,
            'class_idx': int(row['class_idx']),
            'image_idx': int(row['image_idx']),
            'subject': int(row['subject'])
        }

        if self.use_cwt:
            sample['cwt'] = cwt

        if self.use_images:
            feat = self.images[str(sample['class_idx'] + 1)][str(sample['image_idx'] + 1)]
            sample['image'] = torch.from_numpy(feat)
        return sample

