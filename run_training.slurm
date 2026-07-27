#!/bin/bash
#SBATCH --job-name=split_cost
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

#SBATCH --output=/ceph/scratch/bgenca/octagon/agent_training/agent_training/logs/%x_%j.out
#SBATCH --error=/ceph/scratch/bgenca/octagon/agent_training/agent_training/logs/%x_%j.err

module load miniconda/23.10.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /ceph/scratch/bgenca/conda_envs/mlagents

cd /ceph/scratch/bgenca/octagon/agent_training/agent_training

echo "===== Environment ====="
hostname
pwd
which python
/ceph/scratch/bgenca/conda_envs/mlagents/bin/python --version
which pip
which mlagents-learn

/ceph/scratch/bgenca/conda_envs/mlagents/bin/python -c "import numpy; print('numpy OK')"
/ceph/scratch/bgenca/conda_envs/mlagents/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

echo "===== GPU ====="
nvidia-smi

/ceph/scratch/bgenca/conda_envs/mlagents/bin/python run_training_hpc.py