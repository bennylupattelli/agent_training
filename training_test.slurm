#!/bin/bash
#SBATCH --job-name=test_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load miniconda/23.10.0

cd /ceph/scratch/bgenca/octagon/agent_training/agent_training

echo "===== Environment ====="
hostname
pwd
which xvfb-run

/ceph/scratch/bgenca/conda_envs/mlagents/bin/python -c \
"import torch; print(torch.__version__); print(torch.cuda.is_available())"

echo "===== GPU ====="
nvidia-smi

xvfb-run -a -s "-screen 0 1280x1024x24" \
/ceph/scratch/bgenca/conda_envs/mlagents/bin/mlagents-learn \
    /ceph/scratch/bgenca/octagon/template_yaml/SimpleSolo.yaml \
    --env /ceph/scratch/bgenca/octagon/builds/SimpleOctagonAgentSoloLinux/build_260618_01_solo.x86_64 \
    --torch-device cuda:0 \
    --num-envs 1 \
    --run-id test_xvfb \
    --timeout-wait 300 \
    --no-graphics \
    --force