import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable

from dataset import SetDataManager
from meta_template import BaseNet
from scores import compute_otam_scores, compute_proto_scores
from utils import parse_args, get_model

IMAGE_SIZE = 224


def get_embedding(
        embedder: nn.Module,
        x: torch.Tensor,
        y_cuda:bool = True,
        method: str = 'otam',
        temporal_shuffle: bool =False,
        all_pairs_weight:float=0
    ):
    """
    Compute the embedding of the input data
    Args:
        embedder: feature extractor
        x: input data
        y_cuda: whether to move the labels to cuda
        method: which method to use

    Returns:
        scores: the embedding of the input data
        y_query: the labels of the input data
    """
    # Load the input
    x = x.cuda()
    nway, sq, t, c, h, w = x.shape
    x = x.reshape(nway * sq * t, c, h, w)  # 80 images

    if isinstance(embedder, nn.DataParallel):
        n_support = embedder.module.n_support
    else:
        n_support = embedder.n_support
    num_query = sq - n_support

    # Feed to the model and compute the scores
    logits = embedder(x)  # 80 x 2048
    if method == 'otam':
        scores = compute_otam_scores(
            nway,
            num_query,
            n_support,
            logits,
            t,
            sq,
            temporal_shuffle,
            all_pairs_weight=all_pairs_weight
        )  # 5x5
    elif method == 'proto':
        scores = compute_proto_scores(logits, nway, sq, t, n_support, num_query)
    else:
        raise ValueError('Unknown method')

    # Make class indices and return
    if y_cuda:
        y_query = torch.from_numpy(np.repeat(range(nway), num_query))
        y_query = Variable(y_query.cuda())
        return scores, y_query
    return scores, np.repeat(range(nway), num_query)


def test_performance(
    test_model, parameters, loader, epoch, max_acc=0, is_val=True):
    """
    Test the performance of the model
    Args:
        test_model: to be tested
        parameters: parameters (from parse_args)
        loader: data loader
        epoch: which epoch is this (training context)
        max_acc: the maximum accuracy so far (training context)
        is_val: is this a validation or a test

    Returns:
        max_acc: the maximum accuracy so far (training context)
    """
    test_model.eval()
    if not os.path.isdir(parameters.checkpoint_dir):
        os.makedirs(parameters.checkpoint_dir)
    acc_all = []
    iter_num = len(loader)
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            scores, y_query = get_embedding(test_model, x, False, parameters.method)
            _, top_k_labels = scores.data.topk(1, 1, True, True)
            top_k_ind = top_k_labels.cpu().numpy()
            top1_correct = np.sum(top_k_ind[:, 0] == y_query)
            correct_this, count_this = float(top1_correct), len(y_query)
            acc_all.append(correct_this / count_this * 100)
    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    if is_val:
        print('%d Val Acc = %4.2f%% +- %4.2f%%' % (iter_num, float(acc_mean), 1.96 * acc_std / np.sqrt(iter_num)))
        acc = acc_mean
        if acc > max_acc:
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(parameters.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch': epoch, 'state': test_model.state_dict()}, outfile)
        if epoch % parameters.save_freq == 0:
            outfile = os.path.join(parameters.checkpoint_dir, 'epoch{}'.format(epoch))
            torch.save({'epoch': epoch, 'state': test_model.state_dict()}, outfile)
    else:
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, float(acc_mean), 1.96 * acc_std / np.sqrt(iter_num)))
    return max_acc


def train(train_loader, validation_loader, train_model, optimization, parameters):
    """
    Train the model
    Args:
        train_loader: data loader for training
        validation_loader: data loader for validation
        train_model: model to be trained
        optimization: optimization method
        parameters: parameters (from parse_args)

    Returns:
        None
    """
    # Define the optimizer
    if optimization == 'Adam':
        lr = parameters.lr
        print('lr=', lr)
        optimizer = torch.optim.Adam(train_model.parameters(), lr=lr)
    else:
        raise ValueError('Unknown optimization, please define by yourself')

    # Train the model
    max_acc = 0
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(parameters.start_epoch, parameters.stop_epoch):
        train_model.train()
        avg_loss = 0
        for _, (x, _) in enumerate(train_loader):
            scores, y_query = get_embedding(
                train_model,
                x,
                method=parameters.method,
                temporal_shuffle=parameters.temporal_shuffle,
                all_pairs_weight=parameters.all_pairs_weight
            )
            loss = loss_fn(scores, y_query)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()
        print('Epoch {:d} | Loss {:f} | '.format(epoch, avg_loss / float(len(train_loader) + 1)),
              end="" if epoch % 10 == 0 else "\n")
        if epoch % 10 == 0:  # Validation is relatively expensive, so I only do it every 10 epochs
            max_acc = test_performance(train_model, parameters, validation_loader, epoch, max_acc)


def get_loader(parameters, annotation_file, few_shot_params, num_query, mode='train'):
    """
    Get the data loader
    Args:
        parameters: parameters (from parse_args)
        annotation_file: where to find the annotations
        few_shot_params: n_shot, n_way
        num_query: the number of queries

    Returns:
        data loader
    """
    n_episode = 100
    if mode == 'val':
        n_episode = parameters.eval_episode
    elif mode == 'test':
        n_episode = parameters.test_episode
    base_data_manager = SetDataManager(
        IMAGE_SIZE,
        n_query=num_query,
        num_segments=parameters.num_segments,
        n_episode=n_episode,
        **few_shot_params
    )
    return base_data_manager.get_data_loader(
        annotation_file, aug=parameters.train_aug and mode == 'train'
    )


def main():
    """
    Trains/tests proto/otam
    Returns:
        None
    """
    print("Calling meta.py")
    np.random.seed(10)
    params = parse_args('train')
    if params.stop_epoch == -1:
        params.stop_epoch = 400

    # Get the paths to the annotations
    base_file = os.path.join(params.annotation_path, 'train.txt')
    val_file = os.path.join(params.annotation_path, 'val.txt')
    test_file = os.path.join(params.annotation_path, 'test.txt')

    # If test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
    n_query = max(1, int(params.n_query * params.test_n_way / params.train_n_way))

    # Few shot params
    train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
    test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)

    # Get the data loaders
    base_loader = get_loader(params, base_file, train_few_shot_params, n_query)
    val_loader = get_loader(params, val_file, test_few_shot_params, n_query, mode='val')
    test_loader = get_loader(params, test_file, test_few_shot_params, n_query, mode='test')

    # Get the model
    model = BaseNet(get_model(
        params.model,
        params.pretrained,
        params.partial_freeze
    ), **train_few_shot_params).cuda()

    # Determine the checkpoint directory
    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (
        params.work_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    params.checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)
    if params.temporal_shuffle:
        params.checkpoint_dir += '_tshuffle'
    if params.all_pairs_weight:
        params.checkpoint_dir += '_apw'

    # Make the directory if it doesn't exist
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    if params.test_model:
        print(f'Testing model: {params.checkpoint}')
        checkpoint = torch.load(params.checkpoint, map_location=lambda storage, loc: storage.cuda(0))
        checkpoint = checkpoint['state']
        base_dict = {}
        for k, v in list(checkpoint.items()):
            if k.startswith('module'):
                base_dict['.'.join(k.split('.')[1:])] = v
            else:
                base_dict[k] = v
        model.load_state_dict(base_dict)
        test_performance(model, params, test_loader, 0, is_val=False)
    else:
        model = nn.DataParallel(model)  # Load the model

        # Train the model
        train(base_loader, val_loader, model, 'Adam', params)


if __name__ == '__main__':
    main()
