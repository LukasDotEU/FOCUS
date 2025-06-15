from datasets.eegImageNet_dataset import EEGImageNet
from datasets.thingsEEG2_dataset import ThingsEEG2
from models.model_ATMS import ATMS
from models.model_EEGNet import EEGNet
from models.model_EEGChannelNet import EEGChannelNet
from models.model_NiceEEG import NiceEEG

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
		'model_args': {
			'EEGChannelNet': {
				'num_residual_blocks': 4
            },
			'NiceEEG': {
				'clip_centers_file': '../Datasets/EEGImageNet/OnlyUsedImageNet40Images/NICE_clip_center_features.npy',
            },
			'ATMS': {
				'clip_centers_file': '../Datasets/EEGImageNet/OnlyUsedImageNet40Images/ATMS_clip_center_features.npy',
            },
        },
    },
	{
		'name': 'ThingsEEG2',
		'class': ThingsEEG2,
		'eeg_root': '../Datasets/Things-EEG2/Preprocessed_data_250Hz/',
		'images_root': '../Datasets/Things-EEG2/Image_set/',
		'time_steps': 250,
		'num_electrodes': 63,
		'num_classes': 1654,
		'args': {
			'average_reps': False,
        },
		'model_args': {
			'EEGChannelNet': {
				'num_residual_blocks': 3
            },
			'NiceEEG': {
				'clip_centers_file': '../Datasets/Things-EEG2/Image_set/image_set/NICE_clip_center_features.npy',
            },
			'ATMS': {
				'clip_centers_file': '../Datasets/Things-EEG2/Image_set/image_set/ATMS_clip_center_features.npy',
            },
        },
    },
	{
		'name': 'ThingsEEG2Averaged',
		'class': ThingsEEG2,
		'eeg_root': '../Datasets/Things-EEG2/Preprocessed_data_250Hz/',
		'images_root': '../Datasets/Things-EEG2/Image_set/',
		'time_steps': 250,
		'num_electrodes': 63,
		'num_classes': 1654,
		'args': {
			'average_reps': True,
        },
		'model_args': {
			'EEGChannelNet': {
				'num_residual_blocks': 3
            },
			'NiceEEG': {
				'clip_centers_file': '../Datasets/Things-EEG2/Image_set/image_set/NICE_clip_center_features.npy',
            },
			'ATMS': {
				'clip_centers_file': '../Datasets/Things-EEG2/Image_set/image_set/ATMS_clip_center_features.npy',
            },
        },
    },
    # Add more dataset configurations here...
]

# All available model configurations.
MODEL_CONFIGS = [
	# EEGImageNet: 208 -> 40; ThingsEEG2: 624 -> 1654
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
        'use_images': False,
        'epochs': 100,
        'batch_size': 64
    },
	# EEGImageNet: 500 -> 1000 -> 40, ThingsEEG2: 600 -> 1000 -> 1654
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
            # 'num_residual_blocks': 4, # -> Managed Dataset dependant
            'down_kernel': 3,
            'down_stride': 2,
            'learning_rate': 1e-3
        },
        'pretraining': False,
        'use_images': False,
        'epochs': 100,
        'batch_size': 16
    },
	# EEGImageNet: 2960 -> 768, ThingsEEG2: 1440 -> 768
	{
		'name': 'NiceEEG',
		'class': NiceEEG,
		'args': {
			'img_embedding_dim': 768,
            'proj_dim': 768,
			'k': 40,
			'm1': 25,
			'm2': 51,
			's': 5,
			'lr': 2e-4,
			'b1': 0.5,
			'b2': 0.999
        },
		'pretraining': False,
		'use_images': True,
		'dataset_args': {
			'clip_indiviual_feature_file': 'NICE_clip_individual_features.pth'
        },
		'epochs': 200,
		'batch_size': 256 #?
    },
	{
        'name': 'ATMS',
		'class': ATMS,
		'args': {
			'proj_dim': 1024,
			'k': 40,
			'm1': 25,
			'm2': 51,
			's': 5,
			'lr': 3e-4,
            'd_model': 250,
            'dropout': 0.25,
            'n_heads': 4,
            'e_layers': 1,
            'd_ff': 256,
            'activation': 'gelu',
        },
		'pretraining': False,
		'use_images': True,
		'dataset_args': {
			'clip_indiviual_feature_file': 'ATMS_clip_individual_features.pth'
        },
		'epochs': 40, #Check
		'batch_size': 64, # Check, paper says 16 but code 64?
    },
    # Add more model configurations here...
]

# Selected dataset-model combinations to run.
# Use names that match entries in DATASET_CONFIGS and MODEL_CONFIGS.
SELECTED_CONFIGS = [
    {
        'dataset': 'EEGImageNet',
        'model': 'ATMS',
    },
    # Add more combinations as desired...
]