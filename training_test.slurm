#!/bin/bash
#SBATCH --job-name=test_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

module load miniconda/23.10.0

cd /ceph/scratch/bgenca/octagon/agent_training/agent_training

/ceph/scratch/bgenca/conda_envs/mlagents/bin/mlagents-learn \
    /ceph/scratch/bgenca/octagon/template_yaml/SimpleSolo.yaml \
    --env /ceph/scratch/bgenca/octagon/builds/SimpleOctagonAgentSoloLinux/build_260618_01_solo.x86_64 \
    --run-id test \
    --torch-device cuda:0 \
    --num-envs 1 \
    --no-graphics \
    --force