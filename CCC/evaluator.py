import torch
import numpy as np

from tqdm import tqdm
from utils.rerank2 import re_ranking2
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from time import time


@torch.no_grad()
def exact_features(model, data_loader, camera=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()

    features = []
    camera_indexes = []
    data_loader_tqdm = tqdm(data_loader)
    data_loader_tqdm.set_description('    Exact Feature')
    for images, _, cams in data_loader_tqdm:
        embedding = model(images.to(device))
        features.append(embedding.to('cpu'))
        camera_indexes.append(cams)
    features = torch.cat(features)
    camera_indexes = torch.cat(camera_indexes)
    if camera:
        return features, camera_indexes
    else:
        return features


def exact_info(data_loader):
    data_loader.dataset.load_img = False
    pids = [pid for pid, _ in data_loader]
    cams = [cam for _, cam in data_loader]
    pids = torch.cat(pids).cpu().numpy()
    cams = torch.cat(cams).cpu().numpy()
    data_loader.dataset.load_img = True

    return pids, cams


def pairwise_distance(x, y):
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    return dist


class Evaluator(object):
    def __init__(self, model, distance=None):
        super(Evaluator, self).__init__()
        self.model = model

        if distance is None:
            self.distance_func = pairwise_distance
        elif distance == 'Cosine':
            self.distance_func = cosine_distances
        else:
            self.distance_func = euclidean_distances

        self.cmc_scores_list = []
        self.mAP_list = []

    def evaluate(self, query_loader, test_loader, rerank=False):
        query_features = exact_features(self.model, query_loader)#.cpu().numpy()
        test_features = exact_features(self.model, test_loader)#.cpu().numpy()

        # start_time = time()
        distmat = self.distance_func(query_features, test_features)
        if rerank:
            distmat_qq = self.distance_func(query_features, query_features)
            distmat_gg = self.distance_func(test_features, test_features)
            del query_features
            del test_features
            distmat = re_ranking2(distmat, distmat_qq, distmat_gg)
            del distmat_qq
            del distmat_gg
        # print('[init dist]use {}s.'.format(time()-start_time))

        query_pids, query_cams = exact_info(query_loader)
        test_pids, test_cams = exact_info(test_loader)

        cmc_scores, mAP = evaluate_all(distmat, query_pids, test_pids, query_cams, test_cams, cmc_topk=(1, 5, 10))
        self.cmc_scores_list.append(cmc_scores)
        self.mAP_list.append(mAP)

    @property
    def cmc_scores(self):
        return self.cmc_scores_list[-1]

    @property
    def mAP(self):
        return self.mAP_list[-1]


def evaluate_all(distmat, query_pids, test_pids, query_cams, test_cams, cmc_topk=(1, 5, 10)):

    # Compute mean AP
    # start_time = time()
    # 14.913123607635498s.
    mAP = mean_ap(distmat, query_pids, test_pids, query_cams, test_cams)
    # print('[mAP]use {}s.'.format(time()-start_time))

    # Compute cmc_scores
    # start_time = time()
    # 6.472468852996826s.
    cmc_scores = []
    cmc_score = cmc(distmat, query_pids, test_pids, query_cams, test_cams)
    # print('[cmc]use {}s.'.format(time()-start_time))
    for k in cmc_topk:
        cmc_scores.append((k, cmc_score[k-1]))

    return cmc_scores, mAP


def cmc(distmat, query_pids, test_pids, query_cams, test_cams, topk=20):
    m, _ = distmat.shape
    # Ensure numpy array
    query_pids = np.asarray(query_pids)
    test_pids = np.asarray(test_pids)
    query_cams = np.asarray(query_cams)
    test_cams = np.asarray(test_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = np.array(test_pids[indices] == query_pids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((test_pids[indices[i]] != query_pids[i]) |
                 (test_cams[indices[i]] != query_cams[i]))
        if not np.any(matches[i, valid]):
            continue
        index = np.nonzero(matches[i, valid])[0]
        for j, k in enumerate(index):
            if k - j >= topk:
                break
            ret[k - j] += 1
            break
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    return ret.cumsum() / num_valid_queries


def mean_ap(distmat, query_pids, test_pids, query_cams, test_cams):
    m, _ = distmat.shape
    # Ensure numpy array
    query_pids = np.asarray(query_pids)
    test_pids = np.asarray(test_pids)
    query_cams = np.asarray(query_cams)
    test_cams = np.asarray(test_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = np.array(test_pids[indices] == query_pids[:, np.newaxis])
    # Compute AP for each query
    aps = []
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((test_pids[indices[i]] != query_pids[i]) |
                 (test_cams[indices[i]] != query_cams[i]))
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true):
            continue
        aps.append(average_precision_score(y_true, y_score))
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return np.mean(aps)
