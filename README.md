# FOCUS - Framework for Object Classification Using EEG Signals

## What is FOCUS?

FOCUS (Framework for Object Classification Using EEG Signals) is a research-oriented framework designed to standardize and reproduce evaluation in EEG-based object classification tasks. It was developed to accompany our AAAI 2025 Demo Track submission and is intended to facilitate fair benchmarking across datasets and models.

The framework supports multiple neural decoding models, three widely used EEG datasets, and multiple evaluation strategies (within-subject, cross-subject, and stratified CV). It provides a unified interface for running experiments, tracking results, and identifying potential confounds.

📄 This repository accompanies the paper:  
**"Framework for Object Classification Using EEG Signals - FOCUS" (submitted to AAAI 2026 Demo Track)**.

## Project Structure

- **.gitignore**  
  Specifies files and directories to ignore (e.g. `__pycache__/`, `.cache/`).

- **config.py**  
  Central configuration file containing dataset and model configuration parameters ([config.py](config.py)).

- **eeg-object-eval.yml**  
  Conda environment file for the project dependencies ([eeg-object-eval.yml](eeg-object-eval.yml)).

- **evaluate.py**  
  Main script to run training and evaluation experiments. It loads datasets, creates splits, trains models, and outputs the evaluation metrics ([evaluate.py](evaluate.py)).

### Directories

- **datasets/**  
  Contains implementations for dataset precomputing/preprocessing and loading:
  - [`base_dataset.py`](datasets/base_dataset.py): Defines the abstract base class for EEG/Image datasets.
  - [`eegImageNet_dataset.py`](datasets/eegImageNet_dataset.py): Implements the EEGImageNet dataset.
  - [`thingsEEG2_dataset.py`](datasets/thingsEEG2_dataset.py): Implements the ThingsEEG2 dataset.
  - [`kaneshiro_dataset.py`](datasets/kaneshiro_dataset.py): Implements the Kaneshiro dataset.

- **models/**  
  Contains implementations of various neural network models:
  - [`model_base.py`](models/model_base.py): Abstract base model class defining the common interface.
  - [`model_ATMS.py`](models/model_ATMS.py): **[EXPERIMENTAL]** ATMS model adapted from the ATMS repository.
  - [`model_BiLSTM.py`](models/model_BiLSTM.py): BiLSTM model based on BiLSTM-AttGW.
  - [`model_CAWMASASTST.py`](models/model_CAWMASASTST.py): CAWMASASTST model adapted from the CAWMASASTST repository.
  - [`model_CBraMod.py`](models/model_CBraMod.py): CBraMod model adapted from the CBraMod repository. This requires checkpoint weights (`models/model_CBraMod_pretrained_weights.pth`) if using pretrained. These can be found via the origCBraMod repository.
  - [`model_EEGChannelNet.py`](models/model_EEGChannelNet.py): EEGChannelNet model for EEG classification.
  - [`model_EEGClip.py`](models/model_EEGClip.py): EEGClip based on the EEGStyleGAN-ADA implementation.
  - [`model_EEGNet.py`](models/model_EEGNet.py): EEGNet implementation adapted from torcheeg.
  - [`model_NiceEEG.py`](models/model_NiceEEG.py): NICE-EEG model adapted from the NICE-EEG repository.

- **feature_preprocessing/**  
  Contains feature preprocessing scripts:
  - [`ATMS_preprocessing.py`](feature_preprocessing/ATMS_preprocessing.py): Extracts CLIP features for images, CLIP features for labels and saves them.
  - [`CAWMASASTST_eegcwt_preprocessing.py`](feature_preprocessing/CAWMASASTST_eegcwt_preprocessing.py): Computed EEG CWTs at different frequencies and saves them.
  - [`EEGClip_preprocessing.py`](feature_preprocessing/EEGClip_preprocessing.py): Extracts ResNet50 features for images and saves them.
  - [`NiceEEG_preprocessing.py`](feature_preprocessing/NiceEEG_preprocessing.py): Extracts CLIP features for images, averages them per class and saves the center features and individual features.

- **utils/**  
  Contains utility modules for metrics, data splits, and timing:
  - [`metrics.py`](utils/metrics.py): Contains the `Evaluator` class for computing accuracy, F1 score, precision, recall, Cohen’s kappa, and AUC.
  - [`splitGenerator.py`](utils/splitGenerator.py): Provides methods for creating outer and inner splits (per-subject, cross-subject, and stratified CV splits).
  - [`timers.py`](utils/timers.py): Provides a simple context manager (`Timer`) for measuring execution time.

## Setup

1. **Clone the Repository**

   ```sh
   git clone <repo-url>
   cd eeg-object-eval
   ```

2. **Create and Activate the Conda Environment**

   ```sh
   conda env create -f eeg-object-eval.yml
   conda activate eeg-object-eval
   ```

## Datasets

All datasets should be placed in separate folders within a sibling directory of this project named **Datasets**.

This document lists the datasets used in this project along with their sources.

### EEGImageNet

**EEG Data**

- Repository: [EEG Visual Classification](https://github.com/perceivelab/eeg_visual_classification)
- Source (OneDrive): [EEG Visual Data (CVPR 2017)](https://studentiunict-my.sharepoint.com/personal/concetto_spampinato_unict_it/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fconcetto%5Fspampinato%5Funict%5Fit%2FDocuments%2Fsito%5FPeRCeiVe%2Fdatasets%2Feeg%5Fcvpr%5F2017&ga=1)

**ImageNet Images**

Two sources are available for the ImageNet images corresponding to each of the 40 classes:

1. **Complete Set** (containing more than the used images)
   - Repository: [BrainVis](https://github.com/RomGai/BrainVis)
   - Download: [Google Drive Link](https://drive.google.com/file/d/1k3Psdqhl0Saiol4Yauy6eCQK6_-Em05R/view?usp=drive_link)
   
2. **Used Images Subset** (containing only the used images)
   - Repository: [DreamDiffusion](https://github.com/bbaaii/DreamDiffusion)
   - Download: [Google Drive Link](https://drive.google.com/file/d/1y7I9bG1zKYqBM94odcox_eQjnP9HGo9-/view?usp=drive_link)

> **Note:** The image `n03452741_17620.JPEG` was taken from the BrainVis subset because it was corrupted in the DreamDiffusion subset.

### ThingsEEG2

- Repository & Data: [ThingsEEG2 on OSF](https://osf.io/3jk45/)

### Kaneshiro

**EEG Data**

- Original-Version: [Original Version](https://purl.stanford.edu/bq914sc3730)
- Updated-Version: [Object Category EEG Dataset (OCED)](https://exhibits.stanford.edu/data/catalog/tc919dd5388)

**Image Stimuli**

The image stimuli are only available in the updated version embedded in a [pdf file](https://stacks.stanford.edu/file/tc919dd5388/Stimulus%20table.pdf)

## Running the Framework

### Training and Evaluation

Run the main evaluation script to execute the entire training and evaluation pipeline:

```sh
python evaluate.py
```

This script will:
- Load the EEG datasets (for example, EEGImageNet).
- Generate various train/validation/test splits using [`SplitGenerator`](utils/splitGenerator.py).
- Train models (e.g., [`EEGNet`](models/model_EEGNet.py), [`EEGChannelNet`](models/model_EEGChannelNet.py), [`NiceEEG`](models/model_NiceEEG.py), ...) with specified hyperparameters.
- Compute and log metrics using [`Evaluator`](utils/metrics.py) and time each phase with [`Timer`](utils/timers.py).
- Log results through Wandb.

### Running with Slurm

Run the main slurm script to execute the entire training and evaluation pipeline in a detached slurm job:

```sh
sbatch main.sh
```


## Preprocessing

To extract individual dataset trials run the repective dataset file:

```sh
python datasets/eegImageNet_dataset.py
```

To extract features, run one of the feature preprocessing script:

```sh
python feature_preprocessing/NiceEEG_preprocessing.py
```

> **Note**: Make sure to run the scripts from the project root directory due to relative path dependencies.

## Contributing

Contributions are welcome. Please submit pull requests or open issues to report bugs or suggest improvements.