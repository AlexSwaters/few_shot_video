import numpy as np
import torch
import torch.nn.functional as F
from numba import jit
from scipy.spatial.distance import cdist
import random


def shuffle_order(logits: torch.Tensor, shuffle_strength: float = 1):
    """
    Shuffle the temporal order of the logits
    
    Args:
        logits: (nway, sq, t, d)
        shuffle_strength: float, the strength of the shuffle. It's multiplied 
        times the number of frames to get the maximal sum of positional movement.
        A positioinal movement of 1 means that the logits of a frame can be
        moved to the left or right by 1 frame.
    
    Returns:
        shuffled_logits: (nway, sq, t, d)
    """
    t = logits.shape[2]
    shuffle_strength = shuffle_strength * t  # maximal sum of positional movement
    shuffle_strength = random.randint(0, shuffle_strength + 1)
    swaps = []
    while shuffle_strength > 0:
        swap = torch.randint(0, t, (2,)).tolist()
        if swap not in swaps:
            swaps.append(swap)
            shuffle_strength -= abs(swap[0] - swap[1])
    for swap in swaps:
        temp = logits[:, :, swap[0]].clone()
        logits[:, :, swap[0]] = logits[:, :, swap[1]]
        logits[:, :, swap[1]] = temp
    return logits


def compute_all_pairs_weight(z_support, z_query, t):
    """
    Compute the all pairs weight for the OTAM score using cosine similarity.
    
    Args:
        z_support: (t, d)
        z_query: (t, d)
        t: int, number of frames
    
    Returns:
        all_pairs_weight: float, the all pairs weight
    """
    all_pairs_weight = 0
    for i in range(t):
        for j in range(t): # cdist
            pair_dist = np.dot(z_support[i], z_query[j])
            pair_dist /= np.linalg.norm(z_support[i]) * np.linalg.norm(z_query[j])
            all_pairs_weight += pair_dist
    all_pairs_weight /= t * t
    return all_pairs_weight


def compute_otam_scores(
        nway:int, 
        n_query:int, 
        n_support:int, 
        logits:torch.Tensor, 
        t:int, 
        sq:int, 
        use_shuffle=False,
        all_pairs_weight=0
    ):
    """
    Compute the OTAM scores for the given logits.

    Args:
        nway (int): number of support classes
        n_query (int): the number of queries
        n_support (int): the number of support examples per class
        logits (torch.Tensor): the logits
        t (int): the number of segments (or the length of the series of embeddings)
        sq (int): support + query
        use_shuffle (bool, optional): Whether to use shuffle augmentation. Defaults to False.
        all_pairs_weight (int, optional): Contribution of all pairs weight to loss. Defaults to 0.

    Returns:
        torch.Tensor: the scores
    """
    logits = logits.reshape(nway, sq, t, logits.shape[-1])  # 5x2x8x2048
    if (use_shuffle):
        logits = shuffle_order(logits)
    z_support = logits[:, :n_support].reshape(nway * n_support, t, -1)  # 10x8x2048
    z_query = logits[:, n_support:].reshape(nway * n_query, t, -1)  # 10x8x2048

    scores = torch.zeros((nway * n_query, nway * n_support), requires_grad=True).cuda()
    for qi in range(nway * n_query):  # query i
        for si in range(nway * n_support):  # support i
            x = z_support[si].cpu().detach().numpy()
            y = z_query[qi].cpu().detach().numpy()
            
            scores[qi][si] = 0
            if all_pairs_weight > 0:
                scores[qi][si] += compute_all_pairs_weight(x, y, t) * all_pairs_weight

            _, path = dtw(x, y)
            start = np.where(path[0] == 1)[0][0]
            x_index = path[0][start:start + t] - 1  # subtract the extra coordinates
            y_index = path[1][start:start + t]
            scores[qi][si] += F.cosine_similarity(z_support[si][x_index], z_query[qi][y_index]).sum()

            _, path = dtw(y, x)
            start = np.where(path[0] == 1)[0][0]
            x_index = path[0][start:start + t] - 1  # subtract the extra coordinates
            y_index = path[1][start:start + t]
            scores[qi][si] += F.cosine_similarity(z_support[si][y_index], z_query[qi][x_index]).sum()
    return scores.reshape(nway * n_query, nway, n_support).mean(2)  # 5x5


def compute_proto_scores(logits:torch.tensor, nway:int, sq:int, t:int, n_support:int, n_query:int):
    """
    Compute the proto scores for the given logits.

    Args:
        logits (torch.tensor): the logits produced by the model
        nway (int): the number of support classes
        sq (int): support + query
        t (int): the number of segments (or the length of the series of embeddings)
        n_support (int): the number of support examples per class
        n_query (int): the number of queries

    Returns:
        torch.tensor: the scores
    """
    logits = logits.reshape(nway, sq, t, -1)  # 5 x 2 x 8 x 2048
    logits = logits.mean(2)  # average

    z_support = logits[:, :n_support]
    z_query = logits[:, n_support:]

    z_proto = z_support.reshape(nway, n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
    z_query = z_query.reshape(nway * n_query, -1)
    dists = euclidean_dist(z_query, z_proto)
    return -dists


def euclidean_dist(x:torch.Tensor, y:torch.Tensor, normalize:bool=False):
    """
    Compute the euclidean distance between two tensors.

    Args:
        x (torch.Tensor): the first tensor
        y (torch.Tensor): the second tensor
        normalize (bool, optional): Whether to normalize the tensors. Defaults to False.

    Returns:
        torch.Tensor: the distance
    """
    if normalize:
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def dtw(x: np.ndarray, y: np.ndarray):
    """
    Compute the dynamic time warping distance between two sequences.

    Args:
        x (np.array): the first sequence
        y (np.array): the second sequence

    Returns:
        float: the distance
        tuple: the path
    """
    r, c = len(x), len(y)
    r = r + 2
    D0 = np.zeros((r + 1, c + 1))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:]  # view
    D1[1:r - 1, 0:c] = cdist(x, y, 'cosine')
    D0, D1 = dtw_loop(r, c, D0, D1)
    if len(x) == 1:
        path = np.zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), np.zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], path


@jit(nopython=True)
def dtw_loop(r, c, D0, D1):
    """
    Auxiliary function for the dynamic time warping distance.

    Args:
        r (int): rows (the length of the first sequence)
        c (int): columns (the length of the second sequence)
        D0 (np.array): the first distance matrix
        D1 (np.array): the second distance matrix

    Returns:
        np.array: the first distance matrix
        np.array: the second distance matrix
    """
    for i in range(r):  # [0,T+1]
        for j in range(c):  # [0,T-1]
            i_k = min(i + 1, r)
            j_k = min(j + 1, c)
            if i == 0 or i == r - 1:  # first and last
                min_list = [D0[i, j], D0[i_k, j], D0[i, j_k]]
            else:
                min_list = [D0[i, j], D0[i, j_k]]
            D1[i, j] += min(min_list)
    return D0, D1


def _traceback(D):
    """
    Traceback the path.

    Args:
        D (np.array): the distance matrix

    Returns:
        tuple: the path
    """
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = np.argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)
