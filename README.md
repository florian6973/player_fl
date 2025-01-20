# PLayer-FL

This repository contains code for running layer-wise analysis and model evaluations for federated learning systems.

## Getting Started

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
python $DIR/code/layer_pfl_metrics.py --datasets=$DATASETS --federated=$FED
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
- `code/logs/`: Contains execution logs
- `layer_metrics/`: Contains code for layer-wise analysis
- `code/`: Main codebase

