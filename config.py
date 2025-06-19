from datasets.eegImageNet_dataset import EEGImageNet
from datasets.kaneshiro_dataset import Kaneshiro
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
        'eeg_root': '../Datasets/EEGImageNet/',
        'images_root': '../Datasets/EEGImageNet/OnlyUsedImageNet40Images/',
        'time_steps': 440,
        'num_electrodes': 128,
        'num_classes': 40,
        'args': {
			'pth_file': 'eeg_55_95_std.pth',
        },
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
    { # 51840 trials
        'name': 'Kaneshiro',
        'class': Kaneshiro,
        'eeg_root': '../Datasets/Kaneshiro/',
        'images_root': '../Datasets/Kaneshiro/Kaneshiro_images/',
        'time_steps': 651,
        'num_electrodes': 124,
        'num_classes': 6,
        'args': {
			'use_original': False,
        },
		'model_args': {
			'EEGChannelNet': {
				'num_residual_blocks': 3
            },
			'NiceEEG': {
				'clip_centers_file': '../Datasets/Kaneshiro/Kaneshiro_images/NICE_clip_center_features.npy',
            },
			'ATMS': {
				'clip_centers_file': '../Datasets/Kaneshiro/Kaneshiro_images/ATMS_clip_center_features.npy',
            },
        },
    },
    { # 51857 trials
        'name': 'KaneshiroOriginal', # Can't use NiceEEG or ATMS on this one as time_steps not big enough for their convolution
        'class': Kaneshiro,
        'eeg_root': '../Datasets/Kaneshiro/',
        'images_root': '../Datasets/Kaneshiro/Kaneshiro_images/',
        'time_steps': 32,
        'num_electrodes': 124,
        'num_classes': 6,
        'args': {
			'use_original': True,
        },
		'model_args': {
			'EEGChannelNet': {
				'num_residual_blocks': 2
            },
			'NiceEEG': {
				'clip_centers_file': '../Datasets/Kaneshiro/Kaneshiro_images/NICE_clip_center_features.npy',
            },
			'ATMS': {
				'clip_centers_file': '../Datasets/Kaneshiro/Kaneshiro_images/ATMS_clip_center_features.npy',
            },
        },
    },
    # Add more dataset configurations here...
]

# All available model configurations.
MODEL_CONFIGS = [
	# KaneshiroOriginal: 16 -> 6; Kaneshiro: 320 -> 6;EEGImageNet: 208 -> 40; ThingsEEG2: 624 -> 1654
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
	# KaneshiroOriginal (2 residual blocks): 600 -> 1000 -> 6
    # Kaneshiro (3 residual blocks): 7400 -> 1000 -> 6 TODO: Change and recheck as probably overfitting...
	# EEGImageNet (4 residual blocks): 500 -> 1000 -> 40 (Original)
    # ThingsEEG2 (3 residual blocks): 600 -> 1000 -> 1654
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
	# KaneshiroOriginal: NaN; Kaneshiro: 4640 -> 768; EEGImageNet: 2960 -> 768; ThingsEEG2: 1440 -> 768 (Original)
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
            'd_model': 250, # TODO: should this be changed to model specific?
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
        'dataset': 'Kaneshiro',
        'model': 'EEGNet',
    },
    # Add more combinations as desired...
]