import os
import pandas as pd
import torch
import concurrent.futures

from datasets.base_dataset import BaseEEGDataset
from feature_preprocessing import CAWMASASTST_eegcwt_preprocessing

PTH_FILE_DIR = {
    'eeg_55_95_std.pth': 'eegimagenet_individual_pt_55_95',
    'eeg_5_95_std.pth': 'eegimagenet_individual_pt_5_95',
    #'eeg_14_70_std.pth': 'eegimagenet_individual_pt_14_70'
}

LABEL_TEXT_DICT = {'n02106662': 'german shepherd dog',
'n02124075': 'cat ',
'n02281787': 'lycaenid butterfly',
'n02389026': 'sorrel horse',
'n02492035': 'Cebus capucinus',
'n02504458': 'African elephant',
'n02510455': 'panda',
'n02607072': 'anemone fish',
'n02690373': 'airliner',
'n02906734': 'broom',
'n02951358': 'canoe or kayak',
'n02992529': 'cellular telephone',
'n03063599': 'coffee mug',
'n03100240': 'old convertible',
'n03180011': 'desktop computer',
'n03197337': 'digital watch',
'n03272010': 'electric guitar',
'n03272562': 'electric locomotive',
'n03297495': 'espresso maker',
'n03376595': 'folding chair',
'n03445777': 'golf ball',
'n03452741': 'grand piano',
'n03584829': 'smoothing iron',
'n03590841': 'Orange jack-o’-lantern',
'n03709823': 'mailbag',
'n03773504': 'missile',
'n03775071': 'mitten,glove',
'n03792782': 'mountain bike, all-terrain bike',
'n03792972': 'mountain tent',
'n03877472': 'pajama',
'n03888257': 'parachute',
'n03982430': 'pool table, billiard table, snooker table ',
'n04044716': 'radio telescope',
'n04069434': 'eflex camera',
'n04086273': 'revolver, six-shooter',
'n04120489': 'running shoe',
'n07753592': 'banana',
'n07873807': 'pizza',
'n11939491': 'daisy',
'n13054560': 'bolete'
}


def preprocess(pth_file: str, eeg_root: str):
    """
    Reads the .pth at eeg_root/pth_file, splits each sample into its own .pt file under:
      <eeg_root>/eegimagenet_individual_pt_.../
    Also generates:
      - metadata.csv with fields: idx, filename, subject, class_idx, image_idx
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
            'filename': fname,
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
    def __init__(self, eeg_root: str, pth_file: str, images_root: str, sampling_rate: float,
                 use_images: bool = False, images_file: bool = None, use_cwt: bool = False, pre_load: bool = True):
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
        super().__init__(eeg_root=eeg_root, images_root=images_root, sampling_rate=sampling_rate,
                         use_images=use_images, images_file=images_file, use_cwt=use_cwt, pre_load=pre_load)
        self.pth_file = pth_file

        self.samples_dir = os.path.join(self.eeg_root, PTH_FILE_DIR[self.pth_file])
        if not os.path.exists(self.samples_dir):
            print(f"Preprocessing not done before. Preprocesing {self.pth_file} into individual .pt files...")
            preprocess(self.pth_file, self.eeg_root)

        self.metadata = pd.read_csv(os.path.join(self.samples_dir, 'metadata.csv'))
        self.metadata = (self.metadata.sort_values('idx').set_index('idx', drop=False))  # Ensure sorted by idx

        # load saved labels
        self.class_labels = torch.load(os.path.join(self.samples_dir, 'class_labels.pt'))
        self.image_labels = torch.load(os.path.join(self.samples_dir, 'image_labels.pt'))

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
                # subfolders are named after class_labels so can use the class_labels directly
                # this also directly has the right order, as the classlabels in the dataset are not sorted
                if self.images_file.startswith('ATMS'):
                    from feature_preprocessing.ATMS_preprocessing import process_dataset
                    class_text_labels = [LABEL_TEXT_DICT[label] for label in self.class_labels]
                    process_dataset(self.images_root, subfolders=self.class_labels, class_text_labels=class_text_labels)
                elif self.images_file.startswith('NICE'):
                    from feature_preprocessing.NiceEEG_preprocessing import process_dataset
                    process_dataset(self.images_root, subfolders=self.class_labels)
                elif self.images_file.startswith('EEGClip'):
                    from feature_preprocessing.EEGClip_preprocessing import process_dataset
                    process_dataset(self.images_root, subfolders=self.class_labels)
            
            self.images = torch.load(images_file_path, weights_only=False)

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
