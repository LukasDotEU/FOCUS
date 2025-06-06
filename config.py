from datasets.eegImageNet_dataset import EEGImageNet
from models.model_EEGNet import EEGNet
from models.model_EEGChannelNet import EEGChannelNet

# Parameters to add: Epochs, Batch size

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
        },
        'pretraining': False,
        'use_images': False
    },
    {
        'name': 'EEGChannelNet',
        'class': EEGChannelNet,
        'args': {
            'in_channels': 1,
            'temp_channels': 10,
            'out_channels': 50,
            'embedding_size': 1000,
            'temporal_dilation_list': [(1,1),(1,2),(1,4),(1,8),(1,16)],
            'temporal_kernel': (1,33),
            'temporal_stride': (1,2),
            'num_temp_layers': 4,
            'num_spatial_layers': 4,
            'spatial_stride': (2,1),
            'num_residual_blocks': 4,
            'down_kernel': 3,
            'down_stride': 2,
            'learning_rate': 1e-3
        },
        'pretraining': False,
        'use_images': False
    },
    # Add more model configurations here...
]

# Selected dataset-model combinations to run.
# Use names that match entries in DATASET_CONFIGS and MODEL_CONFIGS.
SELECTED_CONFIGS = [
    {
        'dataset': 'EEGImageNet',
        'model': 'EEGChannelNet'
    },
    # Add more combinations as desired...
]