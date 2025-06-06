import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


class SplitGenerator:
    """
    Encapsulates all "outer split" creation logic, plus "inner split" logic:
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
        self.meta = metadata #.copy().reset_index(drop=True)
        self.subject_ids = sorted(self.meta['subject'].unique().tolist())
        self.num_classes = int(self.meta['class_idx'].nunique())
        self.N = len(self.meta)

    # TODO: Change for whole class to use the idx in metadata instead of index of the DataFrame (to be extra sure).
    def get_per_subject_splits(self) -> list[dict]:
        """
        For each subject s, do an 80/20 split within that subject block,
        stratified by class_idx. Return a list of dicts:
            { 'name': str, 'train_idx': [...], 'test_idx': [...] }.
        """
        splits = []
        for sid in self.subject_ids:
            idxs_s = self.meta.index[self.meta['subject'] == sid].tolist()
            if len(idxs_s) < 2:
                print(f"Subject {sid} has fewer than 2 samples → skipping per-subject split.")
                continue

            classes_s = self.meta.loc[idxs_s, 'class_idx'].values
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

            # Only one split, so we can use StratifiedShuffleSplit directly 
            # (would need to use kfold instead if i want to do 10-fold cross validation on each subject as well):
            train_idx_s = []
            test_idx_s = []
            for train_pos, test_pos in sss.split(idxs_s, classes_s):
                train_idx_s = [idxs_s[i] for i in train_pos]
                test_idx_s  = [idxs_s[i] for i in test_pos]

            splits.append({
                'name': f'per_subject_{sid}',
                'train_idx': train_idx_s,
                'test_idx': test_idx_s
            })
            print(f"Created per_subject split for subject {sid}: "
                        f"{len(train_idx_s)} train / {len(test_idx_s)} test")
        return splits

    def get_cross_subject_splits(self) -> list[dict]:
        """
        LOSO: for each subject s, leave out s for test, train on all other subjects.
        Return a list of dicts:
            { 'name': str, 'train_idx': [...], 'test_idx': [...] }.
        """
        splits = []
        all_indices = set(self.meta.index.tolist())
        for sid in self.subject_ids:
            test_idx = self.meta.index[self.meta['subject'] == sid].tolist()
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
        10-fold CV for “all-subjects 80/20” style. We create a combined label 
        combined_lbl = class_idx*1000 + subject so that StratifiedKFold will 
        attempt to distribute each (class,subject) pair evenly across folds.

        Return a list of dicts:
            { 'name': str, 'train_idx': [...], 'test_idx': [...] }.
        """
        # Build combined labels for stratification:
        combined_lbl = (
            self.meta['class_idx'].astype(int) * 1000
            + self.meta['subject'].astype(int)
        ).values

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = []
        for fold_id, (train_pos, test_pos) in enumerate(skf.split(np.arange(self.N), combined_lbl)):
            train_idx = train_pos.tolist()
            test_idx = test_pos.tolist()
            splits.append({
                'name': f'all_subjects_CV_fold_{fold_id}',
                'train_idx': train_idx,
                'test_idx': test_idx
            })
            print(f"Created CV fold {fold_id}: {len(train_idx)} train / {len(test_idx)} test")
        return splits

    def get_inner_split(self, outer_train_idx: list[int], split_name: str) -> tuple[list[int], list[int]]:
        """
        Given an outer_train_idx list and the split_name it belongs to, carve out:
          - train_inner_idx
          - val_idx

        according to the split-type encoded in split_name:
          • If split_name.startswith('per_subject_'): stratify by class_idx within outer_train_idx.
          • If split_name.startswith('cross_subject_LOSO_'): pick exactly one subject from
            outer_train_idx (the subject with fewest samples) to serve as val; the rest → train_inner.
          • If split_name.startswith('all_subjects_CV_fold_'): stratify by combined (class_idx,subject)
            on outer_train_idx with an 80/20 split.

        Returns:
            (train_inner_idx: List[int], val_idx: List[int])
        """
        # PER‐SUBJECT: stratify by class_idx
        if split_name.startswith('per_subject_'):
            classes = self.meta.loc[outer_train_idx, 'class_idx'].values
            if len(outer_train_idx) < 2:
                # Too few samples to split further
                print(f"[{split_name}] outer_train < 2 → no inner split. "
                               f"Returning all {len(outer_train_idx)} as train_inner, 0 as val.")
                return outer_train_idx, []

            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_inner = []
            val_idx = []
            for train_pos, val_pos in sss.split(outer_train_idx, classes):
                train_inner = [outer_train_idx[i] for i in train_pos]
                val_idx = [outer_train_idx[i] for i in val_pos]

            print(f"[{split_name}] Per-subject inner split: {len(train_inner)} train_inner / {len(val_idx)} val")
            return train_inner, val_idx
        
        # TODO: Check this function for correctness, especially the logic for cross-subject splits.
        # CROSS‐SUBJECT (LOSO): pick one subject from outer_train_idx to be val
        elif split_name.startswith('cross_subject_LOSO_'):
            sub_counts = self.meta.loc[outer_train_idx, 'subject'].value_counts().to_dict()
            val_subject = min(sub_counts, key=lambda s: sub_counts[s])
            val_idx = self.meta.index[
                (self.meta['subject'] == val_subject) 
                & (self.meta.index.isin(outer_train_idx))
            ].tolist()
            train_inner = [i for i in outer_train_idx if i not in val_idx]

            print(f"[{split_name}] Cross-subject inner split: subject {val_subject} as val "
                        f"({len(val_idx)} samples), {len(train_inner)} for train_inner")
            return train_inner, val_idx

        # ALL‐SUBJECTS CV: stratify by combined (class_idx,subject)
        elif split_name.startswith('all_subjects_CV_fold_'):
            if len(outer_train_idx) < 2:
                print(f"[{split_name}] outer_train < 2 → no inner split. "
                               f"Returning all {len(outer_train_idx)} as train_inner, 0 as val.")
                return outer_train_idx, []

            subset_meta = self.meta.loc[outer_train_idx]
            combined_lbl = (subset_meta['class_idx'].astype(int) * 1000
                            + subset_meta['subject'].astype(int)).values

            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_inner = []
            val_idx = []
            for train_pos, val_pos in sss.split(outer_train_idx, combined_lbl):
                train_inner = [outer_train_idx[i] for i in train_pos]
                val_idx = [outer_train_idx[i] for i in val_pos]

            print(f"[{split_name}] All‐subjects CV inner split: "
                        f"{len(train_inner)} train_inner / {len(val_idx)} val")
            return train_inner, val_idx

        else:
            raise ValueError(f"Unknown split_name for inner-split: {split_name}")