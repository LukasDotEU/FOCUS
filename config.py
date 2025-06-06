from datasets.eegImageNet_dataset import EEGImageNet
from models.model_EEGNet import EEGNet

# All available dataset configurations.
DATASET_CONFIGS = [
    {
        'name': 'EEGImageNet',
        'class': EEGImageNet,
        'eeg_root': '../Datasets/EEGImageNet/eeg_55_95_std.pth',
        'images_root': '../Datasets/EEGImageNet/OnlyUsedImageNet40Images/',
        'time_steps': 440,
        'num_electrodes': 128,
        'num_classes': 40,
    },
    # Add more dataset configurations here...
]

# All available model configurations.
MODEL_CONFIGS = [
    {
        'name': 'EEGNet',
        'class': EEGNet,
        'args': {
            'F1': 8,
            'F2': 16,
            'D': 2,
            'kernel_1': 64,
            'kernel_2': 16,
            'dropout': 0.25,
            'learning_rate': 1e-3
        }
    },
    # Add more model configurations here...
]

# Selected dataset-model combinations to run.
# Use names that match entries in DATASET_CONFIGS and MODEL_CONFIGS.
SELECTED_CONFIGS = [
    {
        'dataset': 'EEGImageNet',
        'model': 'EEGNet'
    },
    # Add more combinations as desired...
]