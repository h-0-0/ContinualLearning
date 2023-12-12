#!/bin/bash
#
#
#SBATCH --partition=gpu
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=DEBUG
#SBATCH --time=2:00:00
#SBATCH --mem-per-gpu=5G
#SBATCH --account MATH021322

# --nodes=1
# Define executable
export EXE=/bin/hostname

# Change into working directory
cd "${SLURM_SUBMIT_DIR}"

# Execute code
${EXE}

# Do some stuff
echo JOB ID: ${SLURM_JOBID}
echo Working Directory: $(pwd)
echo Start Time: $(date)
nvidia-smi --query-gpu=name --format=csv,noheader

nvidia-smi

# Use this to create virtual env (keeping this here for my reference later on):
# python3 -m venv ./mypyenvb
# We activate the virtual environment
source ../pyvenv/bin/activate

# Regular CL
python main.py --epochs=2 --optimizer_type="SGD" --model_name="resnet18" --n_tasks=5 --learning_rate=0.1

# CL scenario where we stratify the SplitCIFAR10 dataset into two different datasets
# python main.py --epochs=1 --optimizer_type="SGD" --strategy="fixed_replay_stratify" --data2_name="SplitCIFAR10" --batch_ratio=0.8 --percentage=0.8 --model_name="VGG16"

# SSL in batch scenario
# python main.py --ssl_epochs=100 --class_epochs=100 --optimizer_type="SGD" --strategy="ssl" --data_name="SplitCIFAR10" --n_tasks=1 --data2_name="CIFAR10" --model_name="ResNet50" --ssl_batch_size=512

echo End Time: $(date)
