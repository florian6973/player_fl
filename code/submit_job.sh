#!/bin/bash

# List of datasets to process
datasets=("FMNIST"  "EMNIST" "CIFAR" "Sentiment" "ISIC" "mimic" "Heart")
datasets=("Sentiment")

# Options for experiment type
experiment_types=("learning_rate" "evaluation")
experiment_types=("learning_rate")

# Root directory and environment setup
DIR='/gpfs/commons/groups/gursoy_lab/aelhussein/layer_pfl'
ENV_PATH='/gpfs/commons/home/aelhussein/anaconda3/bin/activate'
ENV_NAME='cuda_env_ne1'

mkdir -p logs/outputs logs/errors

for dataset in "${datasets[@]}"; do
    for exp_type in "${experiment_types[@]}"; do
        job_name="${dataset}_${exp_type}"
        cat << EOF > temp_submit_${job_name}.sh
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aelhussein@nygenome.org
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --output=logs/outputs/${job_name}.txt
#SBATCH --error=logs/errors/${job_name}.txt

# Activate the environment
source ${ENV_PATH} ${ENV_NAME}

export PYTHONUNBUFFERED=1
export PYTHON_LOG_DIR="logs/python_logs"
# Run the Python script
python ${DIR}/code/run.py -ds ${dataset} -exp ${exp_type}
EOF

        echo "Submitted job for dataset: ${dataset} and experiment type: ${exp_type}"

        sbatch temp_submit_${job_name}.sh

        rm temp_submit_${job_name}.sh

        sleep 1
    done
done