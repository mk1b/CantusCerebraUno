#!/bin/bash
#SBATCH --job-name=cantus_train
#SBATCH --partition=v100gpu      # Limits the search to ONLY V100 nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1               # One Python process
#SBATCH --cpus-per-task=16       # Matches your dataloader needs
#SBATCH --gpus=1               # Asking for exactly 1 GPU
#SBATCH --mem=64G                
#SBATCH --time=167:59:059          
#SBATCH --output=train_log_%j.out # Auto-names the file with your Job ID!

# Load the toolkit
module load cuda/12.8

# Verify what Slurm gave us (Check your log file for this)
nvidia-smi

source ~/.bashrc
conda activate CantusCerebra

# Matches the 1 GPU requested from Slurm
export TORCH_CUDNN_SDPA_HAS_MATH_BACKEND=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_LAUNCH_BLOCKING=1
python3 -u PretrainUno.py --lr 1e-3 --batch_size 20 --clip_value 7.5 --num_heads 8
