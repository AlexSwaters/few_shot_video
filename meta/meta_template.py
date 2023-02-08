import numpy as np
import torch.nn as nn
from torch.autograd import Variable


class BaseNet(nn.Module):
    """
    Defines a template for meta-learning methods
    """
    def __init__(self, model_func, n_way, n_support):
        super(BaseNet, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = 1  # (change depends on input)
        self.feature = model_func
        self.feat_dim = self.feature.final_feat_dim

    def forward(self, x):
        return self.feature.forward(x)

    def parse_feature(self, x, is_feature):
        """
        Parse features from the embedder into support and query sets
        Args:
            x (torch.Tensor): The embeddings
            is_feature (bool): Whether the input is a feature

        Returns:
            z_support (torch.Tensor): The support set
            z_query (torch.Tensor): The query set
        """
        x = Variable(x.cuda())
        if is_feature:
            z_all = x
        else:
            x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            assert len(x.shape) == 5
            z_all = 0
            for i in range(x.shape[1]):
                z_all += self.feature.forward(x[:, i, :, :, :])
            z_all = z_all / x.shape[1]
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]
        return z_support, z_query

    def correct(self, x):
        scores = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)
        _, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)
