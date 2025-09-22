#!/bin/bash
#SBATCH --account=heilu021
#SBATCH --job-name=FOCUS
#SBATCH --partition=aoraki_gpu_H100
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=128GB
#SBATCH --time=96:00:00
#SBATCH --output=slurm_outputs/%j_%x.out

start_time=$(date +%s)

# Activate environment and run the script
source /projects/sciences/computing/heilu021/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1 # don't add python user site library to path

conda activate FOCUS

python evaluate.py


end_time=$(date +%s)
elapsed_time=$((end_time - start_time))

# Convert elapsed time to days, hours, minutes, and seconds
days=$((elapsed_time / 86400))
hours=$(( (elapsed_time % 86400) / 3600 ))
minutes=$(( (elapsed_time % 3600) / 60 ))
seconds=$((elapsed_time % 60))

# Display the elapsed time in a readable format
echo "Elapsed time: ${days}d ${hours}h ${minutes}m ${seconds}s"