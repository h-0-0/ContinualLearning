#!/bin/bash
#
#
#SBATCH --partition=gpu
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=VGG16_0.8
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --mem=7G
#SBATCH --account MATH021322

# rtx_2080
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

# Use this to create virtual env (keeping this here for my reference later on):
# python3 -m venv ./mypyenvb
# We activate the virtual environment
source ../pyvenv/bin/activate

# Regular CL
# python main.py --epochs=200 --optimizer_type="SGD" --model_name="VGG16" --n_tasks=1

# CL scenario where we stratify the SplitCIFAR10 dataset into two different datasets
python main.py --epochs=15 --optimizer_type="SGD" --strategy="fixed_replay_stratify" --data2_name="SplitCIFAR10" --batch_ratio=0.8 --percentage=0.8 --model_name="VGG16"

# SSL in batch scenario
python main.py --epochs=100 --optimizer_type="SGD" --strategy="ssl" --data_name="SplitImageNet" --data2_name="CIFAR10" --temperature=0.5 --model_name="VGG16" --learning_rate=0.01

echo End Time: $(date)
