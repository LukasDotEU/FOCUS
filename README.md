```markdown
# EEG Object Evaluation

This repository implements several methods for evaluating EEG signals in combination with visual stimuli. The project provides a common interface for building and testing deep neural network models on EEG datasets, with optional integration of associated images.

## Project Structure

- **.gitignore**  
  Specifies files and directories to ignore (e.g. `__pycache__/`).

- **datasets.md [not existent yet]**  
  Contains information about the available datasets ([datasets.md](datasets.md)).

- **eeg-object-eval.yml**  
  Conda environment file for the project dependencies ([eeg-object-eval.yml](eeg-object-eval.yml)).

- **evaluate.py**  
  Main script to run training and evaluation experiments ([evaluate.py](evaluate.py)).

- **datasets/**  
  Contains dataset implementations:
    - `base_dataset.py` – Abstract class for EEG/Image datasets ([datasets/base_dataset.py](datasets/base_dataset.py)).
    - `eegImageNet_dataset.py` – Implementation for EEGImageNet dataset ([datasets/eegImageNet_dataset.py](datasets/eegImageNet_dataset.py)).

- **models/**  
  Contains model implementations:
    - `model_base.py` – Abstract base model class ([models/model_base.py](models/model_base.py)).
    - `model_EEGNet.py` – Implementation of the EEGNet model ([models/model_EEGNet.py](models/model_EEGNet.py)).

- **utils/**  
  Contains utility modules:
    - `metrics.py` – Contains the `Evaluator` class for computing standard metrics ([utils/metrics.py](utils/metrics.py)).
    - `splitGenerator.py` – Provides functionality for dataset splits ([utils/splitGenerator.py](utils/splitGenerator.py)).
    - `timers.py` – Provides a simple context manager for timing ([utils/timers.py](utils/timers.py)).

## Requirements

- Python 3.13
- PyTorch 2.7 with CUDA support (if available)
- Additional dependencies are listed in the [eeg-object-eval.yml](eeg-object-eval.yml) file.

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

## Running Experiments

The main evaluation script (evaluate.py) executes the full training and evaluation pipeline:

```sh
python evaluate.py
```

This script will:

- Load the EEG datasets (such as EEGImageNet).
- Create various train/validation/test splits using the [SplitGenerator](http://_vscodecontentref_/0).
- Train the models (e.g. [EEGNet](http://_vscodecontentref_/1)).
- Evaluate using metrics computed by [Evaluator](http://_vscodecontentref_/2).
- Save results to `evaluation_summary.csv`.

## Contributing

Contributions are welcome. Please submit pull requests or open issues to propose improvements or report bugs.