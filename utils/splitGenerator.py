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

        # Precompute group labels for subject-image pairs
        print(self.meta['image_idx'])
        grp = list(zip(self.meta['subject'], self.meta['image_idx']))
        self.meta['group_id'], _ = pd.factorize(grp)


    # TODO: Change for whole class to use the idx in metadata instead of index of the DataFrame (to be extra sure).
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
        # 1) Build group-level DataFrame (one row per unique image×subject)
        group_df = (
            self.meta
            .loc[:, ['image_idx', 'subject', 'class_idx']]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        # 2) Create a stratification label = class_idx*1000 + subject
        group_df['strat_label'] = group_df['class_idx'].astype(int) * 1000 + group_df['subject'].astype(int)

        # 3) Stratified split on these groups
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = []

        for fold_id, (train_gpos, test_gpos) in enumerate(
            skf.split(group_df, group_df['strat_label'])
        ):
            # Which groups (image, subject) go into train/test for this fold
            train_groups = set(
                tuple(x) for x in group_df.iloc[train_gpos][['image_idx', 'subject']].values
            )
            test_groups = set(
                tuple(x) for x in group_df.iloc[test_gpos][['image_idx', 'subject']].values
            )

            # 4) Map back to full sample indices (including all repetitions)
            is_train = self.meta[['image_idx', 'subject']].apply(tuple, axis=1).isin(train_groups)
            is_test  = self.meta[['image_idx', 'subject']].apply(tuple, axis=1).isin(test_groups)

            train_idx = self.meta.loc[is_train, 'idx'].tolist()
            test_idx  = self.meta.loc[is_test, 'idx'].tolist()

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
            combined_lbl = (
                subset['class_idx'].astype(int) * 1000
                + subset['subject'].astype(int)
            ).values
            if len(combined_lbl) < 2:
                print(f"[{split_name}] outer_train < 2 → no inner split. "
                      f"Returning all {len(outer_train_idx)} as train_inner, 0 as val.")
                return outer_train_idx, []
            sgkf = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=42)
            # This yields 9 splits; we'll grab the first one only which is 10% of original:
            # It's easier this way than to collapse to unique groups,
            # do a single StratifiedShuffleSplit on them an map back up.
            train_ix, val_ix = next(sgkf.split(X=subset['idx'], 
                                               y=combined_lbl, 
                                               groups=subset['group_id'].values))
            train_inner = subset['idx'].iloc[train_ix].tolist()
            val_idx = subset['idx'].iloc[val_ix].tolist()

        else:
            raise ValueError(f"Unknown split: {split_name}")

        
        print(f"{split_name} inner: {len(train_inner)} train_inner / {len(val_idx)} val")
        return train_inner, val_idx
