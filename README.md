# PLayer-FL

This repository contains code for the paper PLayer-FL: A principled approach to personalized layer-wise cross-silo Federated Learning.

## Folder Structure

### `datasets/`
- **`dataset_creator.py`**  
  Run this script first to generate the datasets used in the paper.

- **`dataset_processing.py`**  
  Used internally during the pipeline (not intended to be run directly by the user).

### `layer_metrics/`
Contains scripts to generate the layer metric figures presented in the paper.

### `evaluation/`
Includes scripts to run evaluations on baseline algorithms and **PLayer-FL**.

## Dataset Setup

The project requires several datasets that need to be downloaded and placed in a `data` folder. Below are the required datasets and their sources:

### Directory Structure
```
data/
├── FMNIST/       # Automatically populated
├── EMNIST/       # Automatically populated
├── CIFAR10/       # Automatically populated
├── Heart/         # FLamby heart disease data
├── ISIC/      # FLamby ISIC data
├── mimic_iii/         # MIMIC-III data
└── Sentiment/      # Sentiment140 data
```

### Standard ML Datasets (via PyTorch)
- **Fashion-MNIST (FMNIST)**
  - Available through `torchvision.datasets`
  - Will be downloaded automatically during first run

- **EMNIST**
  - Available through `torchvision.datasets`
  - Will be downloaded automatically during first run

- **CIFAR-10**
  - Available through `torchvision.datasets`
  - Will be downloaded automatically during first run

### Federated Learning Datasets (via FLamby)

- **FED-HEART**
  - **Source**: [FLamby Heart Disease Dataset](https://github.com/owkin/FLamby/tree/main/flamby/datasets/fed_heart_disease)
  - **Installation**: Follow FLamby installation instructions
  - **Directory**: `data/Heart/`
  - **Required Files**:
    - `processed.cleveland.data`: Cleveland data
    - `processed.hungarian.data`: Hungarian data
    - `processed.switzerland.data`: Switzerland data
    - `processed.va.data`: VA data

- **FED-ISIC-2019**
  - **Source**: [FLamby ISIC 2019 Dataset](https://github.com/owkin/FLamby/tree/main/flamby/datasets/fed_isic2019)
  - **Installation**: Follow FLamby installation instructions
  - **Directory**: `data/ISIC/`
  - **Required Files**:
    - `ISIC_2019_Training_GroundTruth.csv`: Ground truth labels from the ISIC 2019 challenge.
    - `ISIC_2019_Training_Metadata_FL.csv`: Associated metadata.
    - `ISIC_2019_Training_Input_preprocessed/`: Folder containing preprocessed JPEG image files.

---

### Healthcare Datasets

- **MIMIC-III**
  - **Source**: [PhysioNet MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/)
  - **Access**: Requires credentialed approval via PhysioNet
  - **Directory**: `data/mimic_iii/`
  - **Required Files**:
    - `ADMISSIONS.csv`: Admissions table
    - `NOTEEVENTS.csv`: Clinical notes table
    - `DIAGNOSIS_ICD.csv`: Diagnosis table
    - `ICUSTAYS.csv`: Stays table

---

### NLP Datasets

- **Sentiment140 (Sent-140)**
  - **Source**: [Kaggle Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
  - **Directory**: `data/sentiment140/`
  - **Required File**:
    - `data/Sentiment/sentiment.zip`: .

---


### Access Requirements
- MIMIC-III requires PhysioNet credentialed access
- Other datasets are publicly available

### Dataset creation
To create the datasets used in the study now run:

```bash
python create_datasets.py [OPTIONS]
```

Available Options
- The default option will create all the datasets
- `--benchmark`: Download and load benchmark image datasets (EMNIST, CIFAR10, FashionMNIST)
- `--isic`: Process the ISIC skin lesion dataset
- `--sentiment`: Process the Twitter sentiment dataset (will unzip sentiment.zip if available)
- `--mimic`: Process the MIMIC-III clinical notes dataset


### Data Processing
For detailed dataset descriptions and preprocessing steps, refer to our paper's methodology section.

## Processes
There are two main processes available:

### 1. Layer Metrics

Located in the `layer_metrics` directory, this codebase reproduces the figures presented in the paper (gradient variance, sample representation, etc.).

#### Running Layer Metrics

Run using the provided SLURM script:

```bash
sbatch script_metrics DIRECTORY DATASETS FED
```

This executes:
```bash
python $DIR/code/layer_metrics/layer_pfl_metrics.py --datasets=$DATASETS --federated=$FED
```

#### Default Parameters
- **Datasets**: Heart, FMNIST, EMNIST, Sentiment, mimic, CIFAR, ISIC
- **Federated Mode**: False

### 2. Model Evaluations

To reproduce the model evaluation results presented in the paper, use `submit_jobs.sh`. This script coordinates both learning rate tuning and final model evaluation.

#### Running Model Evaluations

```bash
bash code/submit_jobs.sh --datasets=DATASETS --exp-types=EXPERIMENT_TYPE --dir=/custom/path
```

Where:
- `EXPERIMENT_TYPE` can be either `evaluation` or `learning_rate`
- For final model evaluation, the best learning rate hyperparameter is selected automatically. This means learning_rate tuning must be done before running evaluation

#### Configuration
- Model configurations can be modified in `configs.py`

## Directory Structure

- `results/`: Contains all output results with subdirectories for `learning_rate` and `evaluation`
- `code/evalation/logs/`: Contains execution logs
- `code/layer_metrics/`: Contains code for layer-wise analysis
- `code/`: Main codebase

