#!/bin/bash
#SBATCH --job-name=gan_test1
#SBATCH --partition=gpu            
#SBATCH --qos=gpu-medium           
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1               
#SBATCH --cpus-per-task=8
#SBATCH --mem=132
#SBATCH --time=23:59:59
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# 1 Load modules/conda env
module purge
module load anaconda3/2021.11
conda init bash
source ~/.bashrc

# activate environment (have to make on adroit first)
conda activate venv 

#export MUJOCO_GL=osmesa
#unset DISPLAY

#export WANDB_API_KEY=028b65afa7cbbb082902c650f03bfd3801495a70
#export WANDB_MODE=offline


python GAN_training.py --gpu_id 0 --model_id gan_test1