# few_shot_video
Repository for submitting few shot video work
## Based on:
- https://openaccess.thecvf.com/content_CVPR_2020/papers/Cao_Few-Shot_Video_Classification_via_Temporal_Alignment_CVPR_2020_paper.pdf
- https://arxiv.org/pdf/2110.12358.pdf
- https://github.com/MCG-NJU/FSL-Video
- https://github.com/wangzehui20/OTAM-Video-via-Temporal-Alignment
## Summary
Ordered temporal alignment has been shown to outperform meta-learning 
baselines. Surprisingly, a simplistic classifier-based baseline with no temporal alignment 
outperforms metric learning with OTAM, largely attributed to the classifier’s ability to 
exploit dropout. In this repository, I implement and evaluate potential 
improvements to the OTAM method to combine the success of classifier-based methods with 
the success of OTAM relative to meta-baselines by attempting to improve the generalization 
of OTAM. I find that temporal order shuffling augmentation and all-pair similarity loss 
contribution do not provide any significant improvement over OTAM and that reducing the 
number of learnable parameters in the OTAM embedder decreases performance.

## File structure
### Scripts
The scripts demonstrate how to call the functionality provided by this repository. You may need 
to change path info. The scripts are intended to be used on the RUG Peregrine cluster.
The module loads will probably not work elsewhere, but you can add the required libraries 
to a virtual environment instead. The script files also activate a virtual environment 
with Torchvision, opencv, numba, tqdm, and a few others. I have a copy of the environment 
in the group folder.
### Baseline plus
baseline_evaluate.py evaluates the performance of a pretrained TSN using Baseline+'s 
fine-tuning methodology. The code for training a TSN can be found here:
https://github.com/liu-zhy/temporal-adaptive-module. 
It is necessary to clone this repository onto your machine, as Basline files require its 
functionality. Note that the actual temporal alignment 
module is not used here. For the pre-training of Baseline and Baseline Plus, most of the 
parameters are the same as the default parameters of TAM (wd=1e-4, batchsize 8x4, 
dropout 0.5). Set lr=1e-3 for all methods without using ImageNet pre-trained weights. 
When using ImageNet pre-trained weights, set lr=1e-4 for classifier-based methods and 
lr=1e-5 for meta-learning methods. The parameters can be specified in yaml files under 
baseline_config. The files in the baseline_plus directory implement 
some required functionality for baseline evaluation.
### Meta
meta.py either trains or evaluates meta-learning methods. This repository currently 
supports the meta-learning baseline and OTAM. Set parameters in meta/utils.py. The meta 
directory also contains supporting files like a template for meta learning methods 
and functions for computing meta-learning method scores.
### Dataset
dataset.py feeds data to the model in training. It includes a few classes, but all are 
geared towards loading frames from a video, providing labels, and feeding them to the model.
Unlike in Baseline+, I do not convert the SSv2 videos to frames, instead coding dataset.py 
to retrieve frames directly from videos.