#!/bin/bash
#SBATCH --job-name=svd_compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4      # Number of GPUs
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

# Activate your conda env
source activate diffusion_env

# example
#srun python run_svd.py \
#    --ckpt-path ./checkpoints/model.pt \
#    --result-path ./results/svd_outputs/ \
#    --data-path ./data/cifar10/ \
#    --time 15 \
#    --batch-size 512 \
#    --score-fn-type symmetrized \
#    --use-lobpcg # we did not use lobpcg

# Run the script using srun
srun python run_svd.py --ckpt-path ./checkpoint.pt --result-path ./results