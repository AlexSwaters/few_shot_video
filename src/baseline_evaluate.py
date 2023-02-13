"""
This file evaluates the Baseline+ model.
"""
import os
import sys

import numpy as np
import torch
import yaml

from baseline_plus.save_features import save_features
from baseline_plus.test import BaselineFinetune, feature_evaluation, gen_cl_file
from dataset import SimpleDataManager

TAM_PATH = '/home/s4284917/temporal-adaptive-module/'
BASE_PATH = '/home/s4284917/few_shot_video/src/'

sys.path.append(TAM_PATH)
from ops.models import TSN

IMAGE_SIZE = 224


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


def get_model(params: dict, print_tsn_info: bool, ckpt_path: str):
    """
    Get the model from checkpoint.

    Args:
        params (dict): Parameters for evaluating Baseline+.
        print_tsn_info (bool): Whether to print TSN info.
        ckpt_path (str): Path to the checkpoint.

    Returns:
        torch.nn.Module: The model.
        int: The final feature dimension.
    """
    # load model
    model = TSN(
        params['base_classes'],
        8,
        'RGB',
        'resnet50',
        tam=params['tam'],
        print_spec=print_tsn_info,
        dropout=0.5 if params['method'] == 'support' else 0.0
    )
    model = model.cuda()
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage.cuda(0))

    for key, value in checkpoint.items():
        if key in ['epoch', 'arch']:
            print(key, value)
    checkpoint = checkpoint['state_dict']
    base_dict = {}
    for k, v in list(checkpoint.items()):
        if k.startswith('module'):
            base_dict['.'.join(k.split('.')[1:])] = v
        else:
            base_dict[k] = v
    model.load_state_dict(base_dict)

    # Determine the final feat dim, make final fc I
    if params['append_classifier']:
        final_feat_dim = 64
    else:
        if params['method'] == 'support':
            model.new_fc = Identity()
        elif params['method'] == 'baseline':
            model.base_model.fc = Identity()  # remove 2048 x 64
        final_feat_dim = 2048

    return model, final_feat_dim


def eval_model(params: dict, final_feat_dim: int, novel_file: str):
    few_shot_params = dict(n_way=params['test_n_way'], n_support=params['n_shot'])
    loss_method = 'softmax' if params['method'] == 'baseline' else 'support'
    model = BaselineFinetune(final_feat_dim=final_feat_dim, loss_type=loss_method, **few_shot_params)
    model = model.cuda()
    acc_all = []
    cl_data_file = gen_cl_file(novel_file)

    iter_num = params['iter_num']
    for i in range(iter_num):
        acc = feature_evaluation(cl_data_file, model, n_query=params['n_query'], **few_shot_params)
        acc_all.append(acc)
    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))


def main():
    """
    Evaluate Baseline+.
    """
    config = BASE_PATH + 'baseline_config/test_baseline.yaml'
    with open(config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    assert params['method'] in ['support', 'baseline']
    split = params['split']
    file = os.path.join(params['data_dir'], '{}.txt'.format(split))
    print(file)
    checkpoint_dir = params['checkpoint']
    novel_file = os.path.join(checkpoint_dir.replace("checkpoints", "features"), split + ".hdf5")
    outfile = novel_file

    print_once = True
    for model_file in os.listdir(checkpoint_dir):
        if os.path.splitext(model_file)[-1] != '.tar':
            continue

        path = os.path.join(checkpoint_dir, model_file)
        print('\n', path)

        dirname = os.path.dirname(outfile)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        datamgr = SimpleDataManager(
            IMAGE_SIZE,
            batch_size=params['batch_size'],
            num_segments=params['num_segments']
        )
        data_loader = datamgr.get_data_loader(data_file=file, aug=False)

        model, final_feat_dim = get_model(params, print_once, path)
        print_once = False

        # save features
        model.eval()
        with torch.no_grad():
            save_features(model, data_loader, outfile, False, True)
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
