# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## General Points
- You are allowed to say 'I don't know'
- Use direct quotes for factual grounding
- Verify claims with citations

## Project Overview
This repo is concerned with batch training and inference of Unity MLAgents models in the 'Octagon' game.
Two other local git repos are important for understanding this project. Octagon is the repo for the Unity implementation
of the Octagon game, both for MLAgents agents and for human players. Octagon_analysis is the repo for analysis, plotting,
and visualisation of the data logged in the Octagon game by both human players and MLAgents RL agents. These repos can be found
in the 'Related Repos' section, and should always be accessible and kept in context.

## Related Repos
- '/home/tom/Unity/Octagon' - Unity project for the Octagon game. Contains the game logic, as well as implementing
netcode for human players (to allow two players to interact across machines), and the code for agent training and inference, which
does not use any netcode and is always local.
- '/home/tom/repos/octagon_analysis' - Python analysis repository for all behavioural and statistical analysis for Unity Octagon,
as well as visualisation and plotting.

## NVIDIA Driver / CUDA on the training machine

`unattended-upgrades` auto-upgrades the NVIDIA driver stack (~monthly). If this happens while
training/inference is running, it swaps the userspace libs without reloading the kernel module,
which breaks CUDA mid-run: `nvidia-smi` reports "Driver/library version mismatch",
`torch.cuda.is_available()` becomes `False`, and runs fail with
`Error 804: forward compatibility was attempted on non supported HW` (fresh runs) or
`Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False`
(loading CUDA checkpoints). A reboot is what reconciles the kernel module with the new libs.

### Auto-updates are disabled via apt holds
NVIDIA packages are pinned with `apt-mark hold` so no apt path (unattended, manual, or
kernel-triggered dkms) upgrades them. To re-apply the holds (e.g. after a fresh install):
```bash
sudo apt-mark hold 'nvidia-*' 'libnvidia-*' 'linux-modules-nvidia-*' 'linux-objects-nvidia-*' 'linux-signatures-nvidia-*'
```
Check what is held with `apt-mark showhold`.

### Manual driver update procedure (run only between run batches)
```bash
# 1. Release the holds
sudo apt-mark unhold 'nvidia-*' 'libnvidia-*' 'linux-modules-nvidia-*' 'linux-objects-nvidia-*' 'linux-signatures-nvidia-*'

# 2. Update the driver
sudo apt update && sudo apt install --only-upgrade 'nvidia-driver-580'

# 3. Reboot — NON-NEGOTIABLE. Reconciles the kernel module with the new userspace libs.
#    Skipping this leaves CUDA in the broken mismatched state.
sudo reboot

# 4. After reboot, re-apply the holds
sudo apt-mark hold 'nvidia-*' 'libnvidia-*' 'linux-modules-nvidia-*' 'linux-objects-nvidia-*' 'linux-signatures-nvidia-*'
```

### Sanity check before launching any run
```bash
nvidia-smi                                                    # versions match, GPU table prints
python -c "import torch; print(torch.cuda.is_available())"    # must print True
```