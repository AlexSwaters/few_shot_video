#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k40:1
#SBATCH --time=30:00:00

module load PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4
module load CUDA/10.1.243-GCC-8.3.0
module load numba/0.47.0-fosscuda-2019b-Python-3.7.4
source /data/s4284917/.envs/fsl/bin/activate

python /home/s4284917/few_shot_video/meta.py --temporal_shuffle True