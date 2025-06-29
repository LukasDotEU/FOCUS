import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, StratifiedGroupKFold

class SplitGenerator:
    """
    Encapsulates outer- and inner-split creation, ensuring no image_idx appears in more than one partition.
      1) Per-subject (80/20 within each subject, stratified on class_idx)
      2) Cross-subject (LOSO: leave one subject out for test, rest→train)
      3) All-subjects 10-fold CV stratified on (class_idx, subject)

      AND:
      A method `get_inner_split(...)` that, given an outer_train_idx and split_name,
      carves out (train_inner_idx, val_idx) according to the same rules.
    """

    def __init__(self, metadata: pd.DataFrame):
        """
        metadata: DataFrame with columns ['idx', 'subject', 'class_idx', 'image_idx'].
                  We assume the DataFrame's index matches the dataset's sample indices (0..N-1).
        """
        self.meta = metadata
        self.subject_ids = sorted(self.meta['subject'].unique().tolist())
        self.N = len(self.meta)

        ## Necessary for k-fold CV
        # 1) factorize (image_idx, subject) into a single group_id
        groups = list(zip(self.meta['image_idx'], self.meta['subject']))
        self.meta['group_id'], _ = pd.factorize(groups)

        # 2) build group_id → all sample idx (all repetitions)
        self.group_to_samples = (self.meta.groupby('group_id')['idx'].apply(list).to_dict())

        # 3) get one row per group and build stratification labels
        repr_rows = (self.meta.drop_duplicates(subset='group_id').set_index('group_id'))
        # unique group IDs
        self.unique_gids = repr_rows.index.to_numpy()
        # strat_label = class_idx * 1000 + subject
        self.strat_labels = (
            repr_rows['class_idx'].astype(int).to_numpy() * 1000
            + repr_rows['subject'].astype(int).to_numpy()
        )

    def get_per_subject_splits(self) -> list[dict]:
        """
        For each subject s, do an 80/20 split within that subject block,
        stratified by class_idx. Return a list of dicts:
            { 'name': str, 'train_idx': [...], 'test_idx': [...] }.
        """
        splits = []
        for sid in self.subject_ids:
            # group by image within subject
            sub_meta = self.meta[self.meta['subject'] == sid]
            img_ids = sub_meta['image_idx'].unique().tolist()
            if len(img_ids) < 2:
                print(f"Subject {sid} has <2 samples → skipping per-subject split.")
                continue

            # label each image by its class (assume consistent)
            img_lbl = [sub_meta[sub_meta['image_idx'] == img]['class_idx'].iat[0] for img in img_ids]
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)

            # Only one split, so we can use StratifiedShuffleSplit directly 
            # (would need to use kfold instead if i want to do 10-fold cross validation on each subject as well):
            train_pos, test_pos = next(sss.split(img_ids, img_lbl))

            train_set = set(img_ids[i] for i in train_pos)
            test_set  = set(img_ids[i] for i in test_pos)

            train_idx = sub_meta[sub_meta['image_idx'].isin(train_set)]['idx'].tolist()
            test_idx  = sub_meta[sub_meta['image_idx'].isin(test_set)]['idx'].tolist()

            splits.append({'name': f'per_subject_{sid}', 'train_idx': train_idx, 'test_idx': test_idx})
            print(f"per_subject_{sid}: {len(train_idx)} train / {len(test_idx)} test")
        return splits

    def get_cross_subject_splits(self) -> list[dict]:
        """
        LOSO: for each subject s, leave out s for test, train on all other subjects.
        Return a list of dicts:
            { 'name': str, 'train_idx': [...], 'test_idx': [...] }.
        """
        splits = []
        all_indices = set(self.meta['idx'].tolist())
        for sid in self.subject_ids:
            test_idx = self.meta[self.meta['subject'] == sid]['idx'].tolist()
            train_idx = list(all_indices.difference(test_idx))
            splits.append({
                'name': f'cross_subject_LOSO_{sid}',
                'train_idx': train_idx,
                'test_idx': test_idx
            })
            print(f"Created LOSO split: subject {sid} as test ({len(test_idx)} samples), "
                  f"{len(train_idx)} samples for training")
        return splits

    def get_stratified_kfold_splits(self, n_splits: int = 10) -> list[dict]:
        """
        10-fold CV: approximate stratification on (class_idx, subject), 
        grouping by unique (image_idx, subject) and then expanding repetitions.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = []

        for fold_id, (train_pos, test_pos) in enumerate(
            skf.split(self.unique_gids, self.strat_labels)
        ):
            train_gids = self.unique_gids[train_pos]
            test_gids  = self.unique_gids[test_pos]

            # 4) Expand back to full sample indices (all 4 reps per group)
            train_idx = [idx for gid in train_gids for idx in self.group_to_samples[gid]]
            test_idx  = [idx for gid in test_gids  for idx in self.group_to_samples[gid]]

            splits.append({
                'name': f'all_subjects_CV_fold_{fold_id}',
                'train_idx': train_idx,
                'test_idx': test_idx
            })
            print(f"all_subjects_CV_fold_{fold_id}: {len(train_idx)} train / {len(test_idx)} test")
        return splits

    def get_inner_split(self, outer_train_idx: list[int], split_name: str):
        # inner splits also group by image
        subset = self.meta.loc[outer_train_idx]
        if split_name.startswith('per_subject_'):
            imgs = subset['image_idx'].unique().tolist()
            if len(imgs) < 2:
                print(f"[{split_name}] outer_train < 2 → no inner split. "
                      f"Returning all {len(outer_train_idx)} as train_inner, 0 as val.")
                return outer_train_idx, []
            lbl = [subset[subset['image_idx'] == img]['class_idx'].iat[0] for img in imgs]
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1/0.9, random_state=42)
            train_pos, val_pos = next(sss.split(imgs, lbl))
            train_imgs = set(imgs[i] for i in train_pos)
            val_set = set(imgs[i] for i in val_pos)

            train_inner = subset[subset['image_idx'].isin(train_imgs)]['idx'].tolist()
            val_idx = subset[subset['image_idx'].isin(val_set)]['idx'].tolist()
            
        elif split_name.startswith('cross_subject_LOSO_'):
            # pick subject with fewest samples in outer_train
            sub_counts = subset['subject'].value_counts()
            val_sub = sub_counts.idxmin()

            val_idx = subset[subset['subject'] == val_sub]['idx'].tolist()
            train_inner = [i for i in subset['idx'].tolist() if i not in val_idx]

        elif split_name.startswith('all_subjects_CV_fold_'):
            # restrict groups to those in outer_train
            outer_meta = self.meta[self.meta['idx'].isin(outer_train_idx)]
            outer_gids = outer_meta['group_id'].unique()

            # their strat labels
            mask = np.isin(self.unique_gids, outer_gids)
            gids = self.unique_gids[mask]
            labels = self.strat_labels[mask]

            # StratifiedShuffleSplit for a single 1/9 validation fold which is 10% of original
            sss = StratifiedShuffleSplit(n_splits=1, test_size=1/9, random_state=42)
            train_pos, val_pos = next(sss.split(gids, labels))

            train_gids = gids[train_pos]
            val_gids   = gids[val_pos]

            # Expand back to all repetitions
            train_inner = [i for gid in train_gids for i in self.group_to_samples[gid]]
            val_idx     = [i for gid in val_gids   for i in self.group_to_samples[gid]]

        else:
            raise ValueError(f"Unknown split: {split_name}")

        
        print(f"{split_name} inner: {len(train_inner)} train_inner / {len(val_idx)} val")
        return train_inner, val_idx
