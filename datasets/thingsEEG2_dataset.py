import os
import numpy as np
import pandas as pd
import torch
import concurrent.futures

from datasets.base_dataset import BaseEEGDataset

def preprocess(eeg_root: str, images_root: str, average_reps: bool = False):
    """
    Reads preprocessed_eeg_training.npy from each subject sub-XX folder,
    splits into individual trial .pt files under:
      <eeg_root>/thingseeg2_individual_pt/
    Also saves metadata.csv, class_labels.pt, image_labels.pt.
    """
    mode = 'averaged' if average_reps else 'multitrials'
    out_dir = os.path.join(eeg_root, f'thingseeg2_individual_pt_{mode}')
    os.makedirs(out_dir, exist_ok=True)

    # load image metadata for labels
    img_meta = np.load(os.path.join(images_root, 'image_metadata.npy'), allow_pickle=True).item()
    image_labels = [f.rsplit('.',1)[0] for f in img_meta['train_img_files']]
    concept_labels = img_meta['train_img_concepts']
    # derive class labels: every 10 images share class
    class_labels = [concept.split('_',1)[1] for i,concept in enumerate(concept_labels) if i % 10 == 0]

    # save label lists
    torch.save(class_labels, os.path.join(out_dir, 'class_labels.pt'))
    torch.save(image_labels, os.path.join(out_dir, 'image_labels.pt'))

    records = []
    trial_idx = 0
    for subfolder in sorted(os.listdir(eeg_root)):
        if not subfolder.startswith('sub-'):
            continue
        subject = int(subfolder.split('-')[1])
        npy_path = os.path.join(eeg_root, subfolder, 'preprocessed_eeg_training.npy')
        data = np.load(npy_path, allow_pickle=True)
        eeg_array = data['preprocessed_eeg_data']  # (n_images=16540, n_reps=4, n_chans=63, n_timesteps=250)
        n_images, n_reps, _, _ = eeg_array.shape

        if average_reps:
            averaged = eeg_array.mean(axis=1)  # shape [n_images, 63, 250]
            for img_idx in range(n_images):
                eeg = torch.from_numpy(averaged[img_idx]).float()
                class_idx = img_idx // 10
                fname = f"trial_{trial_idx:06d}.pt"
                torch.save(eeg, os.path.join(out_dir, fname))
                records.append({
                    'idx': trial_idx,
                    'filename': fname,
                    'subject': subject,
                    'class_idx': class_idx,
                    'image_idx': img_idx
                })
                trial_idx += 1
        else:
            for img_idx in range(n_images):
                for rep in range(n_reps):
                    eeg = torch.from_numpy(eeg_array[img_idx, rep]).float()
                    class_idx = img_idx // 10
                    fname = f"trial_{trial_idx:06d}.pt"
                    torch.save(eeg, os.path.join(out_dir, fname))
                    records.append({
                        'idx': trial_idx,
                        'filename': fname,
                        'subject': subject,
                        'class_idx': class_idx,
                        'image_idx': img_idx
                    })
                    trial_idx += 1
        print(f"processing folder {subfolder}")

    # save metadata
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, 'metadata.csv'), index=False)
    print(f"Saved {trial_idx} trials to {out_dir}")

class ThingsEEG2(BaseEEGDataset):
    def __init__(self, eeg_root: str, images_root: str, average_reps: bool = False,
                 use_images: bool = False, images_file: bool = None, use_cwt: bool = False, pre_load: bool = True):
        """
        Dataset for ThingsEEG2 trials saved via preprocess().

        Expects:
        <eeg_root>/thingseeg2_individual_pt/
          - metadata.csv
          - class_labels.pt
          - image_labels.pt
          - trial_xxxxxx.pt files
        images_root/image_set/... for loading raw images if use_images.
        """
        super().__init__(eeg_root=eeg_root, images_root=images_root, use_images=use_images, images_file=images_file, use_cwt=use_cwt, pre_load=pre_load)
        self.average_reps = average_reps  # If True, average the 4 repetitions of each image
        
        mode = 'averaged' if self.average_reps else 'multitrials'
        self.samples_dir = os.path.join(self.eeg_root, f'thingseeg2_individual_pt_{mode}')
        if not os.path.exists(self.samples_dir):
            print(f"Preprocessing not done before. Preprocesing {mode} ThingsEEG2 into individual .pt files...")
            preprocess(self.eeg_root, self.images_root, average_reps=self.average_reps)

        self.metadata = pd.read_csv(os.path.join(self.samples_dir, 'metadata.csv'))
        self.metadata = (self.metadata.sort_values('idx').set_index('idx', drop=False))  # Ensure sorted by idx

        self.class_labels = torch.load(os.path.join(self.samples_dir, 'class_labels.pt'))
        self.image_labels = torch.load(os.path.join(self.samples_dir, 'image_labels.pt'))

        # depracation compliance when filename column was named filepath
        if 'filepath' in self.metadata.columns:
            self.metadata.rename(columns={'filepath': 'filename'}, inplace=True)
        
        self._filenames = self.metadata['filename'].tolist()
        if self.use_cwt:
            self._cwt_names = ["cwt_" + fp for fp in self._filenames]

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
            self.images = torch.load(os.path.join(self.images_root, 'image_set', self.images_file), weights_only=False)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns a dict:
          {
            'eeg'      : Tensor [ch (63), t],
            'eeg'      : Tensor (if use_cwt=True, else not present),
            'class_idx': int,
            'image_idx': int,
            'subject'  : int,
            'image'    : Tensor [feature_dimension] (if use_images=True, else not present)
          }
        """
        row = self.metadata.iloc[idx]

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
            # lookup using loaded label lists
            class_label = self.class_labels[sample['class_idx']]
            img_label = self.image_labels[sample['image_idx']]
            feat = self.images[class_label][img_label]
            sample['image'] = torch.from_numpy(feat)
        return sample
