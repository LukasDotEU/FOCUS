# EEG Object Evaluation

This repository implements several methods for evaluating EEG signals in combination with visual stimuli. It provides a common interface for building and testing deep neural network models on EEG datasets, with optional integration of associated images. The framework includes dataset loaders, various deep learning models, training/evaluation utilities, and preprocessing scripts.

## Project Structure

- **.gitignore**  
  Specifies files and directories to ignore (e.g. `__pycache__/`, `.cache/`).

- **config.py**  
  Central configuration file containing dataset and model configuration parameters ([config.py](config.py)).

- **datasets.md [non existent yet]**  
  Documentation for the datasets used in the framework ([datasets.md](datasets.md)).

- **eeg-object-eval.yml**  
  Conda environment file for the project dependencies ([eeg-object-eval.yml](eeg-object-eval.yml)).

- **evaluate.py**  
  Main script to run training and evaluation experiments. It loads datasets, creates splits, trains models, and outputs the evaluation metrics ([evaluate.py](evaluate.py)).

### Directories

- **datasets/**  
  Contains implementations for dataset loading:
  - [`base_dataset.py`](datasets/base_dataset.py): Defines the abstract base class for EEG/Image datasets.
  - [`eegImageNet_dataset.py`](datasets/eegImageNet_dataset.py): Implements the EEGImageNet dataset.
#  - [`thingsEEG2_dataset.py`](datasets/thingsEEG2_dataset.py): Implements the ThingsEEG2 dataset.

- **models/**  
  Contains implementations of various neural network models:
  - [`model_base.py`](models/model_base.py): Abstract base model class defining the common interface.
  - [`model_EEGChannelNet.py`](models/model_EEGChannelNet.py): EEGChannelNet model for EEG classification.
  - [`model_EEGNet.py`](models/model_EEGNet.py): EEGNet implementation adapted from torcheeg.
  - [`model_NiceEEG.py`](models/model_NiceEEG.py): NICE-EEG model adapted from the NICE-EEG repository.

- **preprocessing/**  
  Contains preprocessing scripts:
  - [`NiceEEG_preprocessing.py`](preprocessing/NiceEEG_preprocessing.py): Extracts CLIP features for images, averages them per class and saves the center features.

- **utils/**  
  Contains utility modules for metrics, data splits, and timing:
  - [`metrics.py`](utils/metrics.py): Contains the `Evaluator` class for computing accuracy, F1 score, precision, recall, Cohen’s kappa, and AUC.
  - [`splitGenerator.py`](utils/splitGenerator.py): Provides methods for creating outer and inner splits (per-subject, cross-subject, and stratified CV splits).
  - [`timers.py`](utils/timers.py): Provides a simple context manager (`Timer`) for measuring execution time.

## Setup

1. **Clone the Repository**

   ```sh
   git clone <your-repo-url>
   cd eeg-object-eval
   ```

2. **Create and Activate the Conda Environment**

   ```sh
   conda env create -f eeg-object-eval.yml
   conda activate <env-name>
   ```

## Running the Framework

### Training and Evaluation

Run the main evaluation script to execute the entire training and evaluation pipeline:

```sh
python evaluate.py
```

This script will:
- Load the EEG datasets (for example, EEGImageNet).
- Generate various train/validation/test splits using [`SplitGenerator`](utils/splitGenerator.py).
- Train models (e.g., [`EEGNet`](models/model_EEGNet.py), [`EEGChannelNet`](models/model_EEGChannelNet.py), [`NiceEEG`](models/model_NiceEEG.py)) with specified hyperparameters.
- Compute and log metrics using [`Evaluator`](utils/metrics.py) and time each phase with [`Timer`](utils/timers.py).
- Save results to `evaluation_summary.csv`.


## Preprocessing

To extract image features (center features per class) for datasets using CLIP, run the preprocessing script:

```sh
python preprocessing/NiceEEG_preprocessing.py
```

*Note*: Make sure to run this script from the project root directory due to relative path dependencies.

## Contributing

Contributions are welcome. Please submit pull requests or open issues to report bugs or suggest improvements.