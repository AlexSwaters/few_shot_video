"""
This file is responsible for parsing arguments and getting the model.
"""
import argparse

import torch.nn as nn
import torchvision.models as models


def get_model(name: str = 'resnet50', pretrained: bool = True, partial_freeze=False) -> nn.Module:
    """
    Get the model based on parameters.

    Args:
        name (str, optional): name of the model. Defaults to 'resnet50'.
        pretrained (bool, optional): whether to use pretrained model. Defaults to True.
        partial_freeze (bool, optional): whether to freeze the first few layers. Defaults to False.

    Returns:
        nn.Module: the model
    """
    new_model = None
    if name == 'ResNet50':
        new_model = models.resnet50(pretrained=pretrained)
    elif name == 'ResNet34':
        new_model = models.resnet34(pretrained=pretrained)
    new_model.fc = nn.Identity()
    new_model.final_feat_dim = 2048 if name == 'resnet50' else 512
    if partial_freeze:
        for name, layer in new_model.named_children():
            if name == 'layer4':
                break
            for param in layer.parameters():
                param.requires_grad = False
    return new_model


def parse_args(script) -> argparse.Namespace:
    """
    Parse arguments for the script.

    Args:
        script (str): name of the script

    Returns:
        argparse.Namespace: the arguments
    """
    parser = argparse.ArgumentParser(description='few-shot script %s' % (script))
    parser.add_argument('--dataset', default='somethingotam')
    parser.add_argument('--model', default='ResNet50')
    parser.add_argument('--pretrained', default=True)
    parser.add_argument('--partial_freeze', default=False)
    parser.add_argument('--train_n_way', default=5, type=int)
    parser.add_argument('--test_n_way', default=5, type=int)
    parser.add_argument('--n_shot', default=1, type=int)
    parser.add_argument('--train_aug', default=True, type=bool)
    parser.add_argument('--work_dir', default='/home/s4284917/few_shot_video')
    parser.add_argument('--num_segments', default=8, type=int)
    parser.add_argument('--n_query', default=1, type=int)
    parser.add_argument('--eval_freq', default=10, type=int)
    parser.add_argument('--eval_episode', default=1000, type=int)
    parser.add_argument('--test_episode', default=10000, type=int)
    parser.add_argument('--test_model', default=False, type=bool)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--save_freq', default=100, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--stop_epoch', default=-1, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--annotation_path', default='/data/pg-IntSys_Guo/Datasets/FSL_video/smsm-annotations')
    parser.add_argument('--method', default='otam')
    parser.add_argument('--temporal_shuffle', default=False, type=bool)
    parser.add_argument('--all_pairs_weight', default=0.0, type=float)
    return parser.parse_args()
