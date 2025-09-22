from datasets.eegImageNet_dataset import EEGImageNet
from datasets.eood_dataset import EOODDataset
from datasets.kaneshiro_dataset import Kaneshiro
from datasets.thingsEEG2_dataset import ThingsEEG2
from models.model_ATMS import ATMS
from models.model_BiLSTM import BiLSTM
from models.model_CAWMASASTST import CAWMASASTST
from models.model_CBraMod import CBraMod
from models.model_EEGCLIP import EEGClip
from models.model_EEGNet import EEGNet
from models.model_EEGChannelNet import EEGChannelNet
from models.model_NiceEEG import NiceEEG

# All available dataset configurations.
DATASET_CONFIGS = [
    {
        "name": "EEGImageNet",
        "class": EEGImageNet,
        "eeg_root": "../Datasets/EEGImageNet/",
        "images_root": "../Datasets/EEGImageNet/OnlyUsedImageNet40Images/",
        "sampling_rate": 1000,
        "time_steps": 440,
        "num_electrodes": 128,
        "num_classes": 40,
        "args": {
            "pth_file": "eeg_55_95_std.pth",
        },
        "model_args": {
            "NiceEEG": {
                "clip_centers_file": "../Datasets/EEGImageNet/OnlyUsedImageNet40Images/NICE_clip_center_features.npy",
            },
            "ATMS": {
                "clip_centers_file": "../Datasets/EEGImageNet/OnlyUsedImageNet40Images/ATMS_clip_label_features.npy",
            },
            "EEGClip": {"hidden_channels": [128]},
            "BiLSTM": {"hidden_channels": [128]},
        },
    },
    {
        "name": "ThingsEEG2",
        "class": ThingsEEG2,
        "eeg_root": "../Datasets/Things-EEG2/Preprocessed_data_250Hz/",
        "images_root": "../Datasets/Things-EEG2/Image_set/",
        "sampling_rate": 250,
        "time_steps": 250,
        "num_electrodes": 63,
        "num_classes": 1654,
        "args": {
            "average_reps": False,
        },
        "model_args": {
            "EEGChannelNet": {"num_residual_blocks": 3},
            "NiceEEG": {
                "clip_centers_file": "../Datasets/Things-EEG2/Image_set/image_set/NICE_clip_center_features.npy",
            },
            "ATMS": {
                "clip_centers_file": "../Datasets/Things-EEG2/Image_set/image_set/ATMS_clip_label_features.npy",
            },
            "EEGClip": {"hidden_channels": [512, 1024]},  # TODO: Check if something else better
            "BiLSTM": {"hidden_channels": [512, 1024]},
        },
    },
    {
        "name": "ThingsEEG2Averaged",
        "class": ThingsEEG2,
        "eeg_root": "../Datasets/Things-EEG2/Preprocessed_data_250Hz/",
        "images_root": "../Datasets/Things-EEG2/Image_set/",
        "sampling_rate": 250,
        "time_steps": 250,
        "num_electrodes": 63,
        "num_classes": 1654,
        "args": {
            "average_reps": True,
        },
        "model_args": {
            "EEGChannelNet": {"num_residual_blocks": 3},
            "NiceEEG": {
                "clip_centers_file": "../Datasets/Things-EEG2/Image_set/image_set/NICE_clip_center_features.npy",
            },
            "ATMS": {
                "clip_centers_file": "../Datasets/Things-EEG2/Image_set/image_set/ATMS_clip_label_features.npy",
            },
            "EEGClip": {"hidden_channels": [512, 1024]},  # TODO: Check if something else better
            "BiLSTM": {"hidden_channels": [512, 1024]},
        },
    },
    {  # 51840 trials
        "name": "Kaneshiro",
        "class": Kaneshiro,
        "eeg_root": "../Datasets/Kaneshiro/KaneshiroUpdated/",
        "images_root": "../Datasets/Kaneshiro/Kaneshiro_images/",
        "sampling_rate": 1000,
        "time_steps": 651,
        "num_electrodes": 124,
        "num_classes": 6,
        "args": {
            "use_original": False,
        },
        "model_args": {
            "EEGChannelNet": {"num_residual_blocks": 3, "temporal_stride": (1, 14)},
            "NiceEEG": {
                "clip_centers_file": "../Datasets/Kaneshiro/Kaneshiro_images/NICE_clip_center_features.npy",
            },
            "ATMS": {
                "clip_centers_file": "../Datasets/Kaneshiro/Kaneshiro_images/ATMS_clip_label_features.npy",
            },
            "EEGClip": {"hidden_channels": [64]},
            "BiLSTM": {"hidden_channels": [64]},
        },
    },
    {  # 51857 trials
        "name": "KaneshiroOriginal",  # Can't use NiceEEG or ATMS on this one as time_steps not big enough for their convolution
        "class": Kaneshiro,
        "eeg_root": "../Datasets/Kaneshiro/KaneshiroOriginal",
        "images_root": "../Datasets/Kaneshiro/Kaneshiro_images/",
        "sampling_rate": 62.5,
        "time_steps": 32,
        "num_electrodes": 124,
        "num_classes": 6,
        "args": {
            "use_original": True,
        },
        "model_args": {
            "EEGChannelNet": {"num_residual_blocks": 2},
            "NiceEEG": {
                "clip_centers_file": "../Datasets/Kaneshiro/Kaneshiro_images/NICE_clip_center_features.npy",
            },
            "ATMS": {
                "clip_centers_file": "../Datasets/Kaneshiro/Kaneshiro_images/ATMS_clip_label_features.npy",
            },
            "EEGClip": {"hidden_channels": [64]},
            "BiLSTM": {"hidden_channels": [64]},
        },
    },
    {
        "name": "EOODDataset",
        "class": EOODDataset,
        "eeg_root": "../Datasets/EOOD/",
        "images_root": "../Datasets/EOOD/Images/",
        "sampling_rate": 1000,
        "time_steps": 1000,
        "num_electrodes": 64,
        "num_classes": 10,
		"args": {
            "use_sequence": False,
            "baseline_t": [-0.2, 0.0],
            "high_pass": 0.1,
            "low_pass": 100.0,
            "notch_freqs": [50.0],
            "average_reference": True,
            "zscore_norm": True,
        },
        "model_args": {
            "NiceEEG": {
                "clip_centers_file": "../Datasets/EOOD/Images/NICE_clip_center_features.npy",
            },
            "ATMS": {
                "clip_centers_file": "../Datasets/EOOD/Images/ATMS_clip_label_features.npy",
            },
            "EEGClip": {"hidden_channels": [128]},
            "BiLSTM": {"hidden_channels": [128]},
        },
    },
    # Add more dataset configurations here...
]

# All available model configurations.
MODEL_CONFIGS = [
    # KaneshiroOriginal: 16 -> 6; Kaneshiro: 320 -> 6;EEGImageNet: 208 -> 40; ThingsEEG2: 624 -> 1654
    {  # NOTE: Using hyperparameters from "Image classification and reconstruction from low-density EEG"
        # with MaxPooling instead of average. Not using any other model altercations done as not documented enough/unclear.
        "name": "EEGNet",
        "class": EEGNet,
        "args": {
            "F1": 64,
            "F2": 128,  # F1*D=64*2=128
            "D": 2,
            "kernel_1": 64,
            "kernel_2": 16,
            "dropout": 0.25,
            "momentum": 0.1,
            "learning_rate": 1e-3,
        },
        "pretraining": False,
        "use_images": False,
        "use_cwt": False,
        "epochs": 100,
        "batch_size": 64,
    },
    # KaneshiroOriginal (2 residual blocks): 600 -> 1000 -> 6
    # Kaneshiro (3 residual blocks, temporal_stride: (1,14)): 600 -> 1000 -> 6
    # EEGImageNet (4 residual blocks): 500 -> 1000 -> 40 (Original)
    # ThingsEEG2 (3 residual blocks): 600 -> 1000 -> 1654  TODO: Check if embedding size should be increased to 2000
    {
        "name": "EEGChannelNet",
        "class": EEGChannelNet,
        "args": {
            "in_channels": 1,
            "temp_channels": 10,
            "out_channels": 50,
            "embedding_size": 1000,
            "temporal_dilation_list": [(1, 1), (1, 2), (1, 4), (1, 8), (1, 16)],
            "temporal_kernel": (1, 33),
            "temporal_stride": (1, 2),  # overwritten by Kaneshiro dataset config
            "num_temp_layers": 4,
            "num_spatial_layers": 4,
            "spatial_stride": (2, 1),
            "num_residual_blocks": 4,  # overwritten by most dataset configs
            "down_kernel": 3,
            "down_stride": 2,
            "learning_rate": 5e-4,
        },
        "pretraining": False,
        "use_images": False,
        "use_cwt": False,
        "epochs": 100,
        "batch_size": 16,
    },
    # KaneshiroOriginal: NaN; Kaneshiro: 4640 -> 768; EEGImageNet: 2960 -> 768; ThingsEEG2: 1440 -> 768 (Original)
    {
        "name": "NiceEEG",
        "class": NiceEEG,
        "args": {
            "img_embedding_dim": 768,
            "proj_dim": 768,
            "k": 40,
            "m1": 25,
            "m2": 51,
            "s": 5,
            "lr": 2e-4,
            "b1": 0.5,
            "b2": 0.999,
        },
        "pretraining": False,
        "use_images": True,
        "use_cwt": False,
        "dataset_args": {
            "images_individual_feature_file": "NICE_clip_individual_features.pth"
        },
        "epochs": 200,
        "batch_size": 256,  # TODO:?
    },
    {
        "name": "ATMS",
        "class": ATMS,
        "args": {
            "proj_dim": 1024,
            "k": 40,
            "m1": 25,
            "m2": 51,
            "s": 5,
            "lr": 3e-4,
            "d_model": 250,
            "dropout": 0.25,
            "n_heads": 4,
            "e_layers": 1,
            "d_ff": 256,
            "activation": "gelu",
        },
        "pretraining": False,
        "use_images": True,
        "use_cwt": False,
        "dataset_args": {
            "images_individual_feature_file": "ATMS_clip_individual_features.pth"
        },
        "epochs": 40,  # Check
        "batch_size": 64,  # Check, paper says 16 but code 64?
    },
    {
        "name": "CAWMASASTST",
        "class": CAWMASASTST,
        "args": {
            "spectral_channels": 25,
            "lr": 1e-3,
        },
        "pretraining": False,
        "use_images": False,
        "use_cwt": True,
        "epochs": 70,
        "batch_size": 64,
    },
    {
        "name": "CBraMod",
        "class": CBraMod,
        "args": {
            "dropout": 0.1,
            "lr": 1e-4,
            "weight_decay": 5e-2,
            "label_smoothing": 0.1,
            "clip_value": 1,
            "use_pretrained": True,
            "d_model": 200,
            "dim_feedforward": 800,  # 400
            "n_layer": 12,  # 4
            "nhead": 8,  # 4
            "classifier": "all_patch_reps",  # This seems to be original one, although stated params count in paper suggests avgpooling_patch_reps
            "num_patches": None,  # Or number if overlapping patches should be used
        },
        "pretraining": False,
        "use_images": False,
        "use_cwt": False,
        "epochs": 50,
        "batch_size": 64,
    },
    {
        "name": "EEGClip",
        "class": EEGClip,
        "args": {
            "num_layers": 1,
            "pretrain_lr": 3e-4,
            "lr": 1e-4,
        },
        "pretraining": True,
        "use_images": True,
        "use_cwt": False,
        "dataset_args": {
            "images_individual_feature_file": "EEGClip_resnet50_individual_features.pth"
        },
		"pretrain_epochs": 2048,
        "epochs": 250,
        "batch_size": 64,
    },
    {
        "name": "BiLSTM",
        "class": BiLSTM,
        "args": {
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
        "pretraining": False,
        "use_images": False,
        "use_cwt": False,
        "epochs": 50,
        "batch_size": 64,
    },
    # Add more model configurations here...
]

# Selected dataset-model combinations to run.
# Use names that match entries in DATASET_CONFIGS and MODEL_CONFIGS.
SELECTED_CONFIGS = [
    {
        "dataset": "EEGImageNet",
        "model": "ATMS",
    },
    # Add more combinations as desired...
]
