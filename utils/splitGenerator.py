import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, StratifiedGroupKFold

class SplitGenerator:
    """
    Encapsulates outer- and inner-split creation, ensuring no image_idx appears in more than one partition.
      1) Per-subject (90/10 within each subject, stratified on class_idx)
      2) Cross-subject (LOSO: leave one subject out for test, rest→train)
      3) All-subjects 10-fold CV stratified on (class_idx, subject)

      AND:
      A method `get_inner_split(...)` that, given an outer_train_idx and split_name,
      carves out (train_inner_idx, val_idx) according to the same rules.
    """

    def __init__(self, metadata: pd.DataFrame, factorizeBlocks: bool = True):
        """
        metadata: DataFrame with columns ['idx', 'subject', 'class_idx', 'image_idx'].
                  We assume the DataFrame's index matches the dataset's sample indices (0..N-1).
        factorizeBlocks: whether to factorize by (sequence_index) - useful when using multiple small blocks.
        """
        self.meta = metadata
        self.subject_ids = sorted(self.meta['subject'].unique().tolist())
        self.N = len(self.meta)

        ## Necessary for k-fold CV
        # 1) factorize (image_idx, subject) into a single group_id
        if "sequence_index" in self.meta.columns and factorizeBlocks:
            groups = list(zip(self.meta['subject'], self.meta['session_id'], self.meta['sequence_index']))
            print("Factorizing groups by (subject, session_id, sequence_index)")
        else:
            groups = list(zip(self.meta['image_idx'], self.meta['subject']))
            print("Factorizing groups by (image_idx, subject)")
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
        For each subject s, do an 90/10 split within that subject block (with innersplit 80/10/10),
        stratified by class_idx. Return a list of dicts:
            { 'name': str, 'train_idx': [...], 'test_idx': [...] }.

        NOTE: Split at the group-unit level (group_id) so that all samples belonging to the same
        atomic unit (e.g. session_id+sequence_index or image_idx) stay together.
        """
        splits = []
        for sid in self.subject_ids:
            # group by factorized group_id within subject (these are the atomic units)
            sub_meta = self.meta[self.meta['subject'] == sid]
            unit_gids = sub_meta['group_id'].unique().tolist()
            if len(unit_gids) < 2:
                print(f"Subject {sid} has <2 units → skipping per-subject split.")
                continue

            # label each unit by its class (assume consistent within unit)
            unit_lbl = [sub_meta[sub_meta['group_id'] == gid]['class_idx'].iat[0] for gid in unit_gids]
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)

            train_pos, test_pos = next(sss.split(unit_gids, unit_lbl))

            train_gids = [unit_gids[i] for i in train_pos]
            test_gids  = [unit_gids[i] for i in test_pos]

            # expand groups back to sample indices
            train_idx = [idx for gid in train_gids for idx in self.group_to_samples[gid]]
            test_idx  = [idx for gid in test_gids  for idx in self.group_to_samples[gid]]

            splits.append({'name': f'per_subject_{sid}', 'type': 'per_subject', 'train_idx': train_idx, 'test_idx': test_idx})
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
                'type': 'cross_subject',
                'train_idx': train_idx,
                'test_idx': test_idx
            })
            print(f"Created LOSO split: subject {sid} as test ({len(test_idx)} samples), "
                  f"{len(train_idx)} samples for training")
        return splits

    def get_stratified_kfold_splits(self, n_splits: int = 10) -> list[dict]:
        """
        10-fold CV: approximate stratification on (class_idx, subject), 
        grouping by unique group_id (already factorized) and then expanding repetitions.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = []

        for fold_id, (train_pos, test_pos) in enumerate(
            skf.split(self.unique_gids, self.strat_labels)
        ):
            train_gids = self.unique_gids[train_pos]
            test_gids  = self.unique_gids[test_pos]

            # 4) Expand back to full sample indices (all reps per group)
            train_idx = [idx for gid in train_gids for idx in self.group_to_samples[gid]]
            test_idx  = [idx for gid in test_gids  for idx in self.group_to_samples[gid]]

            splits.append({
                'name': f'all_subjects_CV_fold_{fold_id}',
                'type': 'CV',
                'train_idx': train_idx,
                'test_idx': test_idx
            })
            print(f"all_subjects_CV_fold_{fold_id}: {len(train_idx)} train / {len(test_idx)} test")
        return splits

    def get_inner_split(self, outer_train_idx: list[int], split_name: str):
        # inner splits also group by the factorized group_id
        subset = self.meta.loc[outer_train_idx]
        if split_name.startswith('per_subject_'):
            # operate on group units inside the outer_train subset
            unit_gids = subset['group_id'].unique().tolist()
            if len(unit_gids) < 2:
                print(f"[{split_name}] outer_train < 2 → no inner split. "
                      f"Returning all {len(outer_train_idx)} as train_inner, 0 as val.")
                return outer_train_idx, []
            lbl = [subset[subset['group_id'] == gid]['class_idx'].iat[0] for gid in unit_gids]
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1/0.9, random_state=42)
            train_pos, val_pos = next(sss.split(unit_gids, lbl))
            train_gids = [unit_gids[i] for i in train_pos]
            val_gids   = [unit_gids[i] for i in val_pos]

            train_inner = [i for gid in train_gids for i in self.group_to_samples[gid]]
            val_idx = [i for gid in val_gids for i in self.group_to_samples[gid]]
            
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
