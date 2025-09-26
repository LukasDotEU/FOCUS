import os
import h5py
import pandas as pd
import torch

from datasets.base_dataset import BaseEEGDataset

class EOODDataset(BaseEEGDataset):
    def __init__(
        self,
        eeg_root: str,
        images_root: str,
        sampling_rate: float,
        use_images: bool = False,
        images_file: str = None,
        use_cwt: bool = False,
        pre_load: bool = True,
        use_sequence: bool = False,
        sequence_ordinals: list[int] = [1, 2, 3, 4],
        baseline_t: list[float] = [-0.2, 0.0],
        high_pass: float = 0.1,
        low_pass: float = 100.0,
        notch_freqs: list[float] = [50.0],
        resample_freq: float = None,
        average_reference: bool = True,
        zscore_norm: bool = True,
    ):
        """
        PyTorch Dataset for preprocessed EOOD epochs in an HDF5 file.
        
        Expects an HDF5 file with one group per epoch.
        Each group should have:
          - A dataset named 'data' (EEG array with shape = [n_channels, n_times])
          - Attributes that include metadata.
        
        If the preprocessed file does not exist, the preprocessing script
        (preprocess_EOOD.py) will be executed using the provided preprocessing parameters.
        The preprocessed file naming convention is:
          processed_epochs_baseline_<t0>_<t1>_hp<high_pass>_lp<low_pass>_notch<notch1>-<notch2>...[optional _seq].h5
          
        Note:
            eeg_root is the raw data root directory (used as in_raw_dir in the preprocessing call).
        """
        super().__init__(eeg_root=eeg_root, images_root=images_root, sampling_rate=sampling_rate,
                         use_images=use_images, images_file=images_file, use_cwt=use_cwt, pre_load=pre_load)
        # Build a base name using the preprocessing parameters:
        t0, t1 = baseline_t
        notch_str = "-".join(str(nf) for nf in notch_freqs)
        base_name = f"processed_epochs_baseline_{t0}_{t1}_hp{high_pass}_lp{low_pass}_notch{notch_str}"
        # append boolean flags so filename encodes preprocessing choices
        resample_flag = f"resample{resample_freq}" if resample_freq is not None else "noresample"
        avgref_flag = "avgref1" if average_reference else "avgref0"
        zscore_flag = "zscore1" if zscore_norm else "zscore0"
        seq_flag = "seq1" if use_sequence else "seq0"
        base_name = f"{base_name}_{resample_flag}_{avgref_flag}_{zscore_flag}_{seq_flag}.h5"

        self.preproc_path = os.path.join(os.path.dirname(eeg_root), base_name)
        
        if not os.path.exists(self.preproc_path):
            print(f"Preprocessed file not found at {self.preproc_path}. Running preprocessing...")
            # Assume the preprocessing script is accessible from the datasets module.
            from datasets import preprocess_EOOD
            exit_code = preprocess_EOOD.run_preprocessing(
                in_raw_dir=os.path.join(eeg_root,"raws"),
                in_csv_h5=os.path.join(os.path.dirname(eeg_root), "raws", "sequences.h5"),
                out_h5=self.preproc_path,
                baseline_t=baseline_t,
                high_pass=high_pass,
                low_pass=low_pass,
                notch_freqs=notch_freqs,
                resample_freq=resample_freq,
                average_reference=average_reference,
                zscore_norm=zscore_norm,
                use_sequence=use_sequence,
            )
            if exit_code != 0:
                raise RuntimeError("Preprocessing failed; no epochs extracted.")
        
        # Load metadata from preprocessed file
        self.metadata = self._load_metadata()

        # Filter metadata
        self.metadata = self.metadata[self.metadata["type"] == "standard"] # keep only standard trials and no jitter trials

        # Filter metadata to only keep rows with sequence_ordinal equal to 4, if available
        if "sequence_ordinal" in self.metadata.columns:
            self.metadata = self.metadata[self.metadata["sequence_ordinal"].isin(sequence_ordinals)]
        
        # Reset index and add a new 'idx' column representing the new order, then set it as the index.
        self.metadata.reset_index(drop=True, inplace=True)
        self.metadata["idx"] = self.metadata.index

        if self.use_cwt:
            raise NotImplementedError("CWT not implemented for EOODDataset yet.")
        
        if self.pre_load:
            # Preload EEG data for filtered epochs only.
            self._data_cache = []
            with h5py.File(self.preproc_path, "r") as f:
                for epoch_idx in self.metadata["epoch_idx"]:
                    self._data_cache.append(f[f"{epoch_idx}/data"][()])
        
        if self.use_images:
            images_file_path = os.path.join(self.images_root, self.images_file)
            if not os.path.exists(images_file_path):
                subfolders = sorted([e.name for e in os.scandir(self.images_root) 
                                     if e.is_dir() and not 'jitter' in e.name and e.name.endswith('images')])
                if self.images_file.startswith('ATMS'):
                    from feature_preprocessing.ATMS_preprocessing import process_dataset
                    process_dataset(self.images_root, subfolders, class_text_labels=[s.split("_")[0] for s in subfolders])
                elif self.images_file.startswith('NICE'):
                    from feature_preprocessing.NiceEEG_preprocessing import process_dataset
                    process_dataset(self.images_root, subfolders)
                elif self.images_file.startswith('EEGClip'):
                    from feature_preprocessing.EEGClip_preprocessing import process_dataset
                    process_dataset(self.images_root, subfolders)
            
            self.images = torch.load(images_file_path, weights_only=False)

    def _load_metadata(self) -> pd.DataFrame:
        meta_records = []
        with h5py.File(self.preproc_path, "r") as f:
            for idx in sorted(f.keys(), key=lambda x: int(x)):
                grp = f[idx]
                record = {}
                # Extract attributes
                for key, val in grp.attrs.items():
                    if isinstance(val, bytes):
                        val = val.decode("utf-8")
                    record[key] = val
                meta_records.append(record)
        df = pd.DataFrame(meta_records)
        return df

    def _load_epoch(self, epoch_idx: int):
        with h5py.File(self.preproc_path, "r") as f:
            data = f[f"{epoch_idx}/data"][()]
        return data

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]
        if self.pre_load:
            data = self._data_cache[index]
            if self.use_cwt:
                raise NotImplementedError("CWT not implemented for EOODDataset yet.")
        else:
            data = self._load_epoch(row["epoch_idx"])
            if self.use_cwt:
                raise NotImplementedError("CWT not implemented for EOODDataset yet.")
            
        sample = {"eeg": torch.tensor(data)}

        if self.use_cwt:
            raise NotImplementedError("CWT not implemented for EOODDataset yet.")
            sample["cwt"] = torch.tensor(cwt_data)  # Placeholder; implement CWT loading if needed.
        
        if self.use_images:
            class_label = f"{row['class_name']}_{row['class_idx']}_images"
            img_label = f"{row['class_name']}_{row['image_idx']}"
            feat = self.images[class_label][img_label]
            sample['image'] = torch.from_numpy(feat)

        sample.update(row.to_dict())
        return sample
