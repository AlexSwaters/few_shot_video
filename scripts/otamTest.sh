#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k40:1
#SBATCH --time=12:00:00

module load PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4
module load CUDA/10.1.243-GCC-8.3.0
module load numba/0.47.0-fosscuda-2019b-Python-3.7.4
source /data/s4284917/.envs/fsl/bin/activate

python /home/s4284917/few_shot_video/meta.py --test_model True --model ResNet34_pretrained \
    --checkpoint "/home2/s4284917/few_shot_video/checkpoints/somethingotam/ResNet34_pretrained_otam_aug_5way_1shot/best_model.tar"
python /home/s4284917/few_shot_video/meta.py --test_model True \
    --checkpoint "/home2/s4284917/few_shot_video/checkpoints/somethingotam/ResNet50_pretrained_otam_aug_5way_1shot_tshuffle/best_model.tar"
python /home/s4284917/few_shot_video/meta.py --test_model True \
    --checkpoint "/home2/s4284917/few_shot_video/checkpoints/somethingotam/ResNet50_pretrained_otam_aug_5way_1shot_tshuffleHalf/best_model.tar"
python /home/s4284917/few_shot_video/meta.py --test_model True \
    --checkpoint "/home2/s4284917/few_shot_video/checkpoints/somethingotam/ResNet50_pretrained_otam_aug_5way_1shot_apw/best_model.tar"
python /home/s4284917/few_shot_video/meta.py --test_model True \
    --checkpoint "/home2/s4284917/few_shot_video/checkpoints/somethingotam/ResNet50_pretrained_partial_freeze_otam_aug_5way_1shot/best_model.tar"