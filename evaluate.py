import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

# Import dataset classes
from datasets.eegImageNet_dataset import EEGImageNet

# Import model classes
from models.model_EEGNet import EEGNet

# Import utilities
from utils.metrics import Evaluator
from utils.timers import Timer
from utils.splitGenerator import SplitGenerator

# Import configuration.
from config import DATASET_CONFIGS, MODEL_CONFIGS, SELECTED_CONFIGS

NUM_WORKERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_and_evaluate(dataset_conf, model_conf):
    """
    Workflow per (dataset, model):
      1) Load entire dataset → get outer_train_idx, outer_test_idx (per-subject, all-subjects, or LOSO).
      2) From outer_train_idx, carve out a single stratified validation set (VAL_FRACTION of outer_train).
      3) Train on (outer_train/outer_val), validate each epoch on outer_val → early stopping if desired.
         • Keep the weights at the epoch with best validation F1 (or accuracy).
      4) Load those “best” weights, then evaluate once on the outer_test_idx.
      5) Record: total & trainable params, training time, inference time, validation metrics, test metrics.
    """

    results = []
    set_seed(0) # Set a fixed seed for reproducibility

    # 1) Instantiate the full dataset
    DSClass = dataset_conf['class']
    full_dataset = DSClass(
        eeg_root=dataset_conf['eeg_root'],
        images_root=dataset_conf['images_root'],
        use_images=model_conf['use_images']  # Forward use_images to the dataset.
    )

    # Build outer splits
    splitter = SplitGenerator(full_dataset.metadata)
    per_subj_splits = splitter.get_per_subject_splits()
    cross_subj_splits = splitter.get_cross_subject_splits()
    all_subj_cv_splits = splitter.get_stratified_kfold_splits(n_splits=10)

    # Tag each split with a `split_type` for later grouping
    all_outer_splits = per_subj_splits + all_subj_cv_splits + cross_subj_splits

    # Retrieve training parameters from model_conf.
    epochs = model_conf['epochs']
    batch_size = model_conf['batch_size']

    # 3) Loop through each outer split, carve out a single <train/val> and then do final test
    for split in all_outer_splits:
        split_name = split['name']
        outer_train_idx = split['train_idx']
        outer_test_idx = split['test_idx']

        if len(outer_train_idx) < 2:
            print(f"[{split_name}] outer_train has only {len(outer_train_idx)} samples → skipping.")
            continue

        # 2a) carve out inner (train_inner_idx, val_idx) according to split_type via the class method:
        train_inner_idx, val_idx = splitter.get_inner_split(outer_train_idx, split_name)

        # Build DataLoaders with batch_size from model_conf.
        train_loader = DataLoader(
            Subset(full_dataset, train_inner_idx),
            batch_size=batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS
        )
        val_loader = DataLoader(
            Subset(full_dataset, val_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS
        )
        test_loader = DataLoader(
            Subset(full_dataset, outer_test_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS
        )

        # 2b) Instantiate fresh model
        ModelClass = model_conf['class']
        model_args = model_conf['args'].copy()
        model = ModelClass(device=DEVICE, **model_args)
        total_params, trainable_params = model.count_params()
        print(f"[{split_name}] Initialized model '{model_conf['name']}' → total_params={total_params}, trainable_params={trainable_params}")

        # 2c) Training loop (track best-F1 on validation)
        evaluator = Evaluator(average='macro')
        best_val_score = -float('inf')
        best_epoch = -1
        best_state = None

        # Training loop.
        for epoch in range(epochs):
            with Timer() as t_train:
                avg_loss, train_metrics = model.train_one_epoch(train_loader, evaluator)
            train_time = t_train.elapsed

            val_metrics = model.evaluate_on_dataloader(val_loader, evaluator)
            current_f1 = val_metrics['f1']
            if current_f1 > best_val_score:
                best_val_score = current_f1
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            print(
                f"[{split_name}] Epoch {epoch+1} → Loss={avg_loss:.4f} Train_F1={train_metrics['f1']:.4f}, "
                f"Val_F1={current_f1:.4f} (best={best_val_score:.4f} @ epoch {best_epoch+1})"
            )

        # 2d) Reload best weights (if any)
        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"[{split_name}] Reloaded best model from epoch {best_epoch+1} (Val_F1={best_val_score:.4f})")

        # 2e) Evaluate on outer_test_idx
        print(f"[{split_name}] Evaluating on test set ({len(outer_test_idx)} samples)")
        with Timer() as t_test:
            test_metrics = model.evaluate_on_dataloader(test_loader, evaluator)
        test_time = t_test.elapsed

        # Re‐evaluate on val set with best weights
        best_val_metrics = model.evaluate_on_dataloader(val_loader, evaluator)

        # Build result dict:
        row = {
            'dataset':           dataset_conf['name'],
            'model':             model_conf['name'],
            'split_name':        split_name,
            'total_params':      total_params,
            'trainable_params':  trainable_params,
            'best_epoch':        best_epoch + 1,
            'train_time_sec':    train_time,

            # validation metrics at best weights
            'val_accuracy':      best_val_metrics['accuracy'],
            'val_f1':            best_val_metrics['f1'],
            'val_precision':     best_val_metrics['precision'],
            'val_recall':        best_val_metrics['recall'],
            'val_cohen_kappa':   best_val_metrics['cohen_kappa'],
            'val_auc':           best_val_metrics['auc'],

            # test metrics
            'test_accuracy':     test_metrics['accuracy'],
            'test_f1':           test_metrics['f1'],
            'test_precision':    test_metrics['precision'],
            'test_recall':       test_metrics['recall'],
            'test_cohen_kappa':  test_metrics['cohen_kappa'],
            'test_auc':          test_metrics['auc'],
            'test_time_sec':     test_time
        }
        results.append(row)

        print(
            f"[{split_name}] Test results → "
            f"Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}, "
            f"Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, "
            f"Kappa: {test_metrics['cohen_kappa']:.4f}, AUC: {test_metrics['auc']:.4f}, "
            f"Time: {test_time:.4f}s"
        )

        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()

    return results

if __name__ == "__main__":
    all_results = []
    # Iterate over selected configuration combinations.
    for combo in SELECTED_CONFIGS:
        ds_name = combo['dataset']
        model_name = combo['model']

        # Retrieve dataset configuration.
        ds_conf = next(item for item in DATASET_CONFIGS if item['name'] == ds_name)
        # Retrieve model configuration.
        m_conf = next(item for item in MODEL_CONFIGS if item['name'] == model_name)

        m_conf['args']['time_steps'] = ds_conf['time_steps']
        m_conf['args']['num_electrodes'] = ds_conf['num_electrodes']
        m_conf['args']['num_classes'] = ds_conf['num_classes']

        res = train_and_evaluate(ds_conf, m_conf)
        all_results.extend(res)

    df_results = pd.DataFrame(all_results)
    out_csv = "evaluation_summary.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"Evaluation finished. Summary saved to {out_csv}")
