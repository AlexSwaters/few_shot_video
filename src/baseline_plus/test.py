"""
Auxiliary code for testing Baseline+
"""
import random
from abc import abstractmethod

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class SimpleHDF5Dataset:
    """
    Basic torch dataset for HDF5 files
    """
    def __init__(self, file_handle=None):
        if file_handle is None:
            self.f = ''
            self.all_feats_dset = []
            self.all_labels = []
            self.total = 0
        else:
            self.f = file_handle
            self.all_feats_dset = self.f['all_feats'][...]
            self.all_labels = self.f['all_labels'][...]
            self.total = self.f['count'][0]

    def __getitem__(self, i):
        return torch.Tensor(self.all_feats_dset[i, :]), int(self.all_labels[i])

    def __len__(self):
        return self.total


def gen_cl_file(filename: str):
    """
    Produce class label data file which is needed later for evaluation

    Args:
        filename (str): path to the HDF5 file

    Returns:
        dict: class label data file
    """
    with h5py.File(filename, 'r') as f:
        fileset = SimpleHDF5Dataset(f)

    feats = fileset.all_feats_dset
    labels = fileset.all_labels
    while np.sum(feats[-1]) == 0:
        feats = np.delete(feats, -1, axis=0)
        labels = np.delete(labels, -1, axis=0)

    class_list = np.unique(np.array(labels)).tolist()
    inds = range(len(labels))

    cl_data_file = {}
    for cl in class_list:
        cl_data_file[cl] = []
    for ind in inds:
        cl_data_file[labels[ind]].append(feats[ind])

    return cl_data_file


class MetaTemplate(nn.Module):
    """
    Template for Baseline+ adaptation
    """
    def __init__(self, n_way, n_support, change_way=True):
        super(MetaTemplate, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = -1  # (change depends on input)
        self.change_way = change_way  # some methods allow different_way classification during training and test

    @abstractmethod
    def set_forward(self, x, is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    def parse_feature(self, x):
        """
        Convert features to support and query sets

        Args:
            x (torch.Tensor): features

        Returns:
            z_support (torch.Tensor): support features
            z_query (torch.Tensor): query features
        """
        z_all = Variable(x.cuda())
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]
        return z_support, z_query

    @abstractmethod
    def set_forward_adaptation(self, x):
        """
        Adapt the TSN by appending a linear classifier and fine-tuning it. Produce scores
        and accuracy for queries
        """
        z_support, z_query = self.parse_feature(x)
        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))
        y_support = Variable(y_support.cuda())
        linear_clf = nn.Linear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.cuda()
        set_optimizer = torch.optim.SGD(
            linear_clf.parameters(),
            lr=0.01,
            momentum=0.9,
            dampening=0.9,
            weight_decay=0.001
        )

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()

        batch_size = 4
        support_size = self.n_way * self.n_support
        for _ in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]
                scores = linear_clf(z_batch)
                loss = loss_function(scores, y_batch)
                loss.backward()
                set_optimizer.step()

        scores = linear_clf(z_query)
        return scores


class BaselineFinetune(MetaTemplate):
    """
    Wrapper for fine-tuning the TSN from TAM to the few-shot context
    """
    def __init__(self, n_way, n_support, loss_type="softmax", final_feat_dim=2048):
        super(BaselineFinetune, self).__init__(n_way, n_support)
        self.loss_type = loss_type
        self.feat_dim = final_feat_dim

    def forward(self, x):
        raise ValueError('Baseline predict on pretrained feature and do not support finetune backbone')

    def set_forward(self, x, is_feature=True):
        return self.set_forward_adaptation(x, is_feature)

    def __get_linear_clf(self, z_support):
        """
        Get the linear classifier that will be attached to the end of the model
        """
        linear_clf = nn.Linear(self.feat_dim, self.n_way)
        if self.loss_type == 'support':
            linear_clf = nn.Linear(self.feat_dim, self.n_way)
            if z_support.shape[0] == self.n_way:
                feature_relu = nn.ReLU()
                init_weight = nn.functional.normalize(feature_relu(z_support), 2, 1)  # L2 norm
                init_bias = torch.zeros(self.n_way).float()
                linear_clf.weight.data = init_weight
                linear_clf.bias.data = init_bias
            elif z_support.shape[0] == self.n_way * self.n_support:
                z_mean = torch.mean(torch.reshape(z_support, (self.n_way, self.n_support, -1)), dim=1)
                feature_relu = nn.ReLU()
                init_weight = nn.functional.normalize(feature_relu(z_mean), 2, 1)  # L2 norm
                init_bias = torch.zeros(self.n_way).float()
                linear_clf.weight.data = init_weight
                linear_clf.bias.data = init_bias
        return linear_clf.cuda()

    def set_forward_adaptation(self, x, temporal_aug=False):
        """
        Adapt the TSN by appending a linear classifier and fine-tuning it. Produce scores
        and accuracy for queries. This overrides the default implementation in MetaTemplate.
        """
        z_support, z_query = self.parse_feature(x)
        t = x.shape[-2] if temporal_aug else 1
        z_support = z_support.contiguous().view(self.n_way * self.n_support * t, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query * t, -1)
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support * t))
        y_support = Variable(y_support.cuda())

        linear_clf = self.__get_linear_clf(z_support)
        set_optimizer = torch.optim.SGD(
            linear_clf.parameters(),
            lr=1e-2,
            momentum=0.9,
            dampening=0.9,
            weight_decay=0.001
        )
        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()

        batch_size = 4
        if temporal_aug:
            support_size = self.n_way * self.n_support * t
        else:
            support_size = self.n_way * self.n_support

        for _ in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]
                if self.loss_type == 'support':
                    z_relu = nn.ReLU().cuda()
                    z_batch = nn.functional.normalize(z_relu(z_batch), 2, 1)  # L2 norm

                scores = linear_clf(z_batch)
                loss = loss_function(scores, y_batch)
                loss.backward()
                set_optimizer.step()

        scores = linear_clf(z_query)
        pred = scores.data.cpu().numpy().argmax(axis=1)
        y = np.repeat(range(self.n_way), self.n_query)
        acc = np.mean(pred == y) * 100
        return scores, acc

    def set_forward_loss(self, _):
        raise ValueError('Baseline predict on pretrained feature and do not support finetune backbone')


def feature_evaluation(cl_data_file, model, n_way=5, n_support=5, n_query=1, temporal_aug=False):
    """
    Evaluate the features produced by the model and return the accuracy (predicting which class queries
    correspond to)
    """
    class_list = cl_data_file.keys()
    select_class = random.sample(class_list, n_way)
    z_all = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in range(n_support + n_query)])  # stack each batch
    z_all = torch.from_numpy(np.array(z_all))
    model.n_query = n_query
    _, acc = model.set_forward_adaptation(z_all, temporal_aug=temporal_aug)
    return acc
