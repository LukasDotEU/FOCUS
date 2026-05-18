# FOCUS - Framework for Object Classification Using EEG Signals

## What is FOCUS?

FOCUS (Framework for Object Classification Using EEG Signals) is a research-oriented framework designed to standardize and reproduce evaluation in EEG-based object classification tasks.

The framework supports multiple neural decoding models, three widely used EEG datasets, and multiple evaluation strategies (within-subject, cross-subject, and stratified CV). It provides a unified interface for running experiments, tracking results, and identifying potential confounds.

## Project Structure

- **.gitignore**  
  Specifies files and directories to ignore (e.g. `__pycache__/`, `.cache/`).

- **config.py**  
  Central configuration file containing dataset and model configuration parameters ([config.py](config.py)).

- **FOCUS.yml**  
  Conda environment file for the project dependencies ([FOCUS.yml](FOCUS.yml)).

- **evaluate.py**  
  Main script to run training and evaluation experiments. It loads datasets, creates splits, trains models, and outputs the evaluation metrics ([evaluate.py](evaluate.py)).

### Directories

- **datasets/**  
  Contains implementations for dataset precomputing/preprocessing and loading:
  - [`base_dataset.py`](datasets/base_dataset.py): Defines the abstract base class for EEG/Image datasets.
  - [`eegImageNet_dataset.py`](datasets/eegImageNet_dataset.py): Implements the EEGImageNet dataset.
  - [`thingsEEG2_dataset.py`](datasets/thingsEEG2_dataset.py): Implements the ThingsEEG2 dataset.
  - [`kaneshiro_dataset.py`](datasets/kaneshiro_dataset.py): Implements the Kaneshiro dataset.
  - [`eegCORD_dataset.py`](datasets/eegCORD_dataset.py) & [`preprocess_eegCORD.py`](datasets/preprocess_eegCORD.py): Implements the EEG-CORD dataset which is currently being prepared to be published (includes the separate preprocessing).

- **models/**  
  Contains implementations of various neural network models:
  - [`model_base.py`](models/model_base.py): Abstract base model class defining the common interface.
  - [`model_ATMS.py`](models/model_ATMS.py): ATMS model adapted from the ATMS repository.
  - [`model_BiLSTM.py`](models/model_BiLSTM.py): BiLSTM model based on BiLSTM-AttGW.
  - [`model_CAWMASASTST.py`](models/model_CAWMASASTST.py): CAWMASASTST model adapted from the CAWMASASTST repository.
  - [`model_CBraMod.py`](models/model_CBraMod.py): CBraMod model adapted from the CBraMod repository. This requires checkpoint weights (`models/model_CBraMod_pretrained_weights.pth`) if using pretrained. These can be found via the origCBraMod repository.
  - [`model_EEGChannelNet.py`](models/model_EEGChannelNet.py): EEGChannelNet model for EEG classification.
  - [`model_EEGClip.py`](models/model_EEGClip.py): EEGClip based on the EEGStyleGAN-ADA implementation.
  - [`model_EEGNet.py`](models/model_EEGNet.py): EEGNet implementation adapted from torcheeg.
  - [`model_NiceEEG.py`](models/model_NiceEEG.py): NICE-EEG model adapted from the NICE-EEG repository.
  - [`model_TRIAGE_EEG.py`](models/model_TRIAGE_EEG.py): Experimental TRIAGE-EEG model for which no publication is currently planned.

- **feature_preprocessing/**  
  Contains feature preprocessing scripts needed for certain models:
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
   cd FOCUS
   ```

2. **Create and Activate the Conda Environment**

   ```sh
   conda env create -f FOCUS.yml
   conda activate FOCUS
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

### EEG-CORD

**In process to be submitted for publication**

This is our own collected dataset which is currently being prepared for publication submission.

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

## Model Implementation Differences

Due to inconsistencies between code and paper descriptions, dataset-dependent input dimensions, and classifier output requirements, we made several adaptations to the original models.

- **EEGNet**: We used the [original version](https://dx.doi.org/10.1088/1741-2552/aace8c), applying hyperparameters and MaxPooling (instead of AveragePooling) as in [Guenther et al. (2024)](https://www.nature.com/articles/s41598-024-66228-1). Other unclear modifications in that paper were omitted. The final linear layer projects to the number of classes.

- **NiceEEG**: We corrected the image projection module, which previously returned the input unchanged instead of applying its layers. This fix led to improved performance in an earlier evaluation of ours, raising the 200-class zero-shot accuracy from 13.80% to 16.95% compared to the original paper.

- **EEGChannelNet**: We employed three residual blocks for both the ThingsEEG2 and Kaneshiro datasets instead of the original four. For the Kaneshiro dataset, the temporal stride was adjusted from `(1, 2)` to `(1, 14)` to maintain a consistent feature size prior to the MLP. The final linear layer projects the features to the corresponding number of classes. The joint learning variant was not implemented due to missing source code. Because training proved unstable even on the filtered EEGImageNet dataset on which the model was originally developed, we reduced the learning rate from 1e-3 to 5e-4. A potential cause of the instability could be the use of different split types, as the original implementation employed a fixed across-all-subjects split.

- **EEGClip**: We used precomputed image features to avoid redundant computation. For the classifier MLP, the hidden layer size was 128 for EEGImageNet, 64 for Kaneshiro, and two hidden layers with sizes 512 followed by 1024 for ThingsEEG2. Pretraining was run for 2048 epochs and fine-tuning for 250, following the original code.

- **ATMS**: The codebase was difficult to navigate, with undocumented data embedding components not mentioned in the paper. While the paper mentions subject-specific tokens, the code uses additional subject-specific embedding layers as well as a shared token and embedding layer for unknown subjects. We trained the shared layer and token on a randomly selected 10% of the data (when trained using CV or LOSO splits) for the other 90% of the data the subject-specific layer and token were used. Furthermore, the batch size was not clearly specified in the [paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/ba5f1233efa77787ff9ec015877dbd1f-Paper-Conference.pdf), where values of either 16 or 1024 could be inferred. Inspection of the released code indicated a batch size of 64, which we adopted.

  **Note:** ATMS makes use of class labels. In the ThingsEEG2 dataset, there are instances such as `Bow1`, `Bow2`, and `Bow3`, each referring to different meanings of the word *bow* (hair bow, shooting bow, and present bow, respectively). Since there was no straightforward automatic fix, we retained the original class names as-is.

- **CAWMASASTST**: Since no transformation code to the time-frequency domain was provided, we implemented it ourselves, assisted by a [forked repository](https://github.com/busiqiao/CAW-MASA-STST), which also fixed a bug in the model’s forward method. This fork served as the base for our implementation and training, including hyperparameters.

- **BiLSTM**: Implemented as a bidirectional LSTM followed by a MLP. The MLP architecture mirrors that of EEGClip. Hyperparameters were adopted from [Zheng and Chen (2021)](https://www.sciencedirect.com/science/article/pii/S174680942030313X).

- **CBraMod**: We used the pretrained model and weights provided. As it was not originally applied to our datasets, we adapted the classifier layer to fit the EEG input and the output dimensions. We selected the `all_patch_reps` classifier variant, as it appeared to be the one used in the original implementation, although the parameter count in the paper would suggest `avgpooling_patch_reps`. We did not modify the model’s architecture otherwise.

## Preprocessing

Preprocessing for individual datasets is automatically being done when being run for the first time. 

> **Note**: Make sure to run the scripts from the project root directory due to relative path dependencies.

## Contributing

Contributions are welcome. Please submit pull requests or open issues to report bugs or suggest improvements.