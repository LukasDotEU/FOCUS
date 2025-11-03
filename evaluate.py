import copy
import os
import json
import argparse
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Subset
import wandb

# Import base model class
from datasets.base_dataset import BaseEEGDataset
from models.model_base import BaseModel

# Import utilities
from utils.metrics import Evaluator
from utils.timers import Timer
from utils.splitGenerator import SplitGenerator

# Import configuration.
from config import DATASET_CONFIGS, MODEL_CONFIGS, SELECTED_CONFIGS

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_and_evaluate(dataset_conf: dict, model_conf: dict, save_dir: str, testing: bool = False):
    """
    Workflow per (dataset, model):
      1) Load entire dataset → get outer_train_idx, outer_test_idx (per-subject, all-subjects, or LOSO).
      2) From outer_train_idx, carve out a single stratified validation set (VAL_FRACTION of outer_train).
      3) Train on (outer_train/outer_val), validate each epoch on outer_val → early stopping if desired.
         • Keep the weights at the epoch with best validation F1 (or accuracy).
      4) Load those “best” weights, then evaluate once on the outer_test_idx.
      5) Record: total & trainable params, training time, inference time, validation metrics, test metrics.
    """

    set_seed(0) # Set a fixed seed for reproducibility

    # Prepare run directory for this dataset-model-split
    split_tag = f"{dataset_conf['name']}_{model_conf['name']}"

    # Setup run-specific save directory
    run_dir = os.path.join(save_dir, split_tag)
    os.makedirs(run_dir, exist_ok=True)
    # Save run configuration
    config_data = {
        'dataset_conf': {k: v for k, v in dataset_conf.items() if k != 'class'},
        'model_conf': {k: v for k, v in model_conf.items() if k != 'class'},
    }
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config_data, f, indent=2)

    # Instantiate full dataset
    DSClass = dataset_conf['class']
    dataset_args = model_conf.get('dataset_args', None)
    images_file = (dataset_args.get('images_individual_feature_file', None) if dataset_args else None)
    ds_kwargs = {
        'eeg_root': dataset_conf['eeg_root'],
        'images_root': dataset_conf['images_root'],
        'use_images': model_conf['use_images'],
        'use_cwt': model_conf['use_cwt'],
        'sampling_rate': dataset_conf['sampling_rate'],
        'images_file': images_file,
        **dataset_conf.get('args', {})
    }

    pre_load = False if model_conf['name'] == "CAWMASASTST" else True
    full_dataset: BaseEEGDataset = DSClass(pre_load=pre_load, **ds_kwargs)
    num_workers = 0 if pre_load else 4

    factorizeBlocks = dataset_conf.get('factorizeBlocks', True)
    # Build outer splits
    splitter = SplitGenerator(full_dataset.metadata, factorizeBlocks)
    per_subj_splits = splitter.get_per_subject_splits()
    cross_subj_splits = splitter.get_cross_subject_splits()
    all_subj_cv_splits = splitter.get_stratified_kfold_splits(n_splits=10)
    all_outer_splits = per_subj_splits + all_subj_cv_splits + cross_subj_splits

    # Retrieve training parameters from model_conf
    epochs = model_conf['epochs']
    batch_size = model_conf['batch_size']

    # Loop through each outer split, carve out a single <train/val>, then do train and testing
    for split in all_outer_splits:
        split_name = split['name']
        outer_train_idx = split['train_idx']
        outer_test_idx = split['test_idx']

        if len(outer_train_idx) < 2:
            print(f"[{split_name}] outer_train has only {len(outer_train_idx)} samples → skipping.", flush=True)
            continue

        # Inner split for validation
        train_inner_idx, val_idx = splitter.get_inner_split(outer_train_idx, split_name)

        train_loader = DataLoader(Subset(full_dataset, train_inner_idx),
                                  batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(Subset(full_dataset, val_idx),
                                batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(Subset(full_dataset, outer_test_idx),
                                 batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        # Instantiate model
        ModelClass = model_conf['class']
        model_name = model_conf['name']
        model_args = copy.deepcopy(model_conf['args'])

        # Give ATMS model info about number of subjects it's being trained with for subject specific logic
        if model_name == "ATMS":
            if split_name.startswith("per_subject"):
                num_subjects = 1
            elif split_name.startswith("cross_subject"):
                num_subjects = len(np.unique(full_dataset.metadata['subject'])) - 2
            elif split_name.startswith("all_subjects"):
                num_subjects = len(np.unique(full_dataset.metadata['subject']))
            else:
                Exception("Splitname not known.")
            model_args['num_subjects'] = num_subjects
        # inject epochs and train_size into CBraMod for lr_scheduler
        elif model_name == "CBraMod":
            model_args['epochs'] = epochs
            model_args['train_size'] = len(train_loader)

        model: BaseModel = ModelClass(device=DEVICE, **model_args)
        total_params, trainable_params = model.count_params()
        print(f"[{split_name}] Initialized model '{model_name}' with '{ds_conf['name']}'", flush=True)
        print(f"[{split_name}] total_params={total_params}, trainable_params={trainable_params}", flush=True)

        # Prepare wandb config
        wandb_config = {
            "dataset": dataset_conf['name'],
            "dataset_conf": {k: v for k, v in dataset_conf.items() if k != 'class'},
            "model": model_conf['name'],
            "model_conf": {k: v for k, v in model_conf.items() if k != 'class'},
            "split_name": split_name,
            "split_type": split['type'],
            "total_params": total_params,
            "trainable_params": trainable_params,
        }
        run_name = f"{dataset_conf['name']}_{model_conf['name']}_{split_name}"
        wandb.init(
            project="FOCUS",  #FOCUS-EEGImageNet
            name=run_name,
            config=wandb_config,
            mode="disabled" if testing else "online",
        )

        if model_conf.get('pretraining', False) and model_conf.get('pretrain_epochs', None):
            print(f"[{split_name}] Pretraining model for {model_conf['pretrain_epochs']} epochs.", flush=True)
            for pretrain_epoch in range(model_conf['pretrain_epochs']):
                pretrain_loss = model.pretrain_one_epoch(train_loader)
                print(f"[{split_name}] Pretrain Epoch {pretrain_epoch+1}/{model_conf['pretrain_epochs']}. loss={pretrain_loss:.4f}", flush=True)
                wandb.log({"pretrain_epoch": pretrain_epoch + 1, "pretrain_loss": pretrain_loss})

        evaluator = Evaluator(average='macro')
        best_val_score = -float('inf')
        best_epoch = -1
        best_state = None
        best_val_train_metrics = None
        best_val_metrics = None

        # Training loop (track best-F1 on validation)
        for epoch in range(epochs):
            # Training
            with Timer() as t_train:
                avg_train_loss, train_metrics = model.train_one_epoch(train_loader, evaluator)
            train_time = t_train.elapsed

            # Validation
            avg_val_loss, val_metrics = model.evaluate_on_dataloader(val_loader, evaluator)
            current_f1 = val_metrics['f1']
            if current_f1 > best_val_score:
                best_val_score = current_f1
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_val_train_metrics = train_metrics
                best_val_metrics = val_metrics

            print(
                f"[{split_name}] Epoch {epoch+1}: loss={avg_train_loss:.4f}, train_F1={train_metrics['f1']:.4f}, "
                f"val_F1={current_f1:.4f} (best={best_val_score:.4f} @ epoch {best_epoch+1})", flush=True
            )

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_accuracy": train_metrics['accuracy'],
                "train_balanced_acc": train_metrics['balanced_acc'],
                "train_f1": train_metrics['f1'],
                "train_precision": train_metrics['precision'],
                "train_recall": train_metrics['recall'],
                "train_cohen_kappa": train_metrics['cohen_kappa'],
                "train_time_sec": train_time,
                "val_loss": avg_val_loss,
                "val_accuracy": val_metrics['accuracy'],
                "val_balanced_acc": val_metrics['balanced_acc'],
                "val_f1": val_metrics['f1'],
                "val_precision": val_metrics['precision'],
                "val_recall": val_metrics['recall'],
                "val_cohen_kappa": val_metrics['cohen_kappa'],
            })

        # Reload best weights
        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"[{split_name}] Reloaded best model from epoch {best_epoch+1} (val_f1={best_val_score:.4f})", flush=True)

        # Final evaluation on test set
        with Timer() as t_test:
            avg_test_loss, test_metrics = model.evaluate_on_dataloader(test_loader, evaluator, test_pred=True)
        test_time = t_test.elapsed

        # Prepare test metrics as a wandb.Table
        test_table = wandb.Table(
            columns=[
                "model_name", "dataset_name", "split_type", "split_name",
                "test_loss", "test_accuracy", "test_balanced_acc", "test_f1", "test_precision", "test_recall",
                "test_cohen_kappa", "test_confusion_matrix", "test_auc", "test_time_sec", "best_epoch"
            ],
            data=[[
                model_conf['name'],
                dataset_conf['name'],
                split['type'],
                split_name,
                avg_test_loss,
                test_metrics['accuracy'],
                test_metrics['balanced_acc'],
                test_metrics['f1'],
                test_metrics['precision'],
                test_metrics['recall'],
                test_metrics['cohen_kappa'],
                test_metrics['confusion_matrix'],
                test_metrics['auc'],
                test_time,
                best_epoch + 1,
            ]]
        )
        wandb.log({"test_metrics_table": test_table})
        wandb.log({"best_epoch": best_epoch + 1})
        wandb.finish()

        # Checkpoint: save from best epoch
        checkpoint = {
            'epoch': best_epoch+1,
            'model_state_dict': best_state,
            'metrics': {
                'train': best_val_train_metrics,
                'val': best_val_metrics,
                'test': test_metrics
            }
        }
        ckpt_path = os.path.join(run_dir, f'{split_name}.pt')
        if not testing: # Change to always save if desired
            torch.save(checkpoint, ckpt_path)

        print(
            f"[{split_name}] Test results → "
            f"Acc: {test_metrics['accuracy']:.4f}, Balanced Acc: {test_metrics['balanced_acc']:.4f}, "
            f"F1: {test_metrics['f1']:.4f}, Precision: {test_metrics['precision']:.4f}, "
            f"Recall: {test_metrics['recall']:.4f}, Kappa: {test_metrics['cohen_kappa']:.4f}, "
            f"AUC: {test_metrics['auc']:.4f}, Time: {test_time:.4f}s", flush=True
        )

        # Clean up
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate models with checkpointing')
    parser.add_argument('--testing', action='store_true', help='Save to test_model_weights instead of model_weights')
    args = parser.parse_args()

    if args.testing:
        print("Running in testing mode.", flush=True)

    base_dir = 'testing_model_weights' if args.testing else 'model_weights'
    os.makedirs(base_dir, exist_ok=True)

    # Iterate over selected configuration combinations.
    for combo in SELECTED_CONFIGS:
        # Retrieve dataset configuration.
        ds_conf = next(d for d in DATASET_CONFIGS if d['name'] == combo['dataset'])

        # Retrieve model configuration.
        m_conf = next(m for m in MODEL_CONFIGS if m['name'] == combo['model'])
        # Update model args with dataset specifics
        m_conf['args'].update({
            'time_steps': ds_conf['time_steps'],
            'num_electrodes': ds_conf['num_electrodes'],
            'num_classes': ds_conf['num_classes'],
            **ds_conf.get('model_args', {}).get(combo['model'], {})
        })

        train_and_evaluate(ds_conf, m_conf, base_dir, args.testing)

    print(f"Evaluation finished.", flush=True)
