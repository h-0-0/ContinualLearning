#!/bin/bash
#
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_2080:1
#SBATCH --job-name=Test
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --mem=5G
#SBATCH --account MATH021322


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
python main.py --epochs=10 --optimizer_type="Adam"

# CL scenario where we stratify the SplitCIFAR10 dataset into two different datasets
# python main.py --epochs=10 --optimizer_type="Adam" --strategy="fixed_replay_stratify" --data2_name="SplitCIFAR10" --batch_ratio=0.9 --percentage=0.9 

echo End Time: $(date)
