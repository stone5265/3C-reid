#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

import time
import numpy as np
import faiss

import torch
import torch.nn.functional as F

try:
    from utils.faiss_utils import search_index_pytorch, search_raw_array_pytorch, index_init_gpu, index_init_cpu
except:
    from faiss_utils import search_index_pytorch, search_raw_array_pytorch, index_init_gpu, index_init_cpu


def k_reciprocal_neigh(initial_rank, k1):
    # forward_k_neigh_index = initial_rank[i, :k1+1]
    # backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1+1]
    # fi = np.where(backward_k_neigh_index == i)[0]
    num_probes = initial_rank.shape[0]

    forward_k_neigh_index = initial_rank[:, :k1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1]
    probe_indexes = np.arange(num_probes).reshape(-1, 1).repeat(k1 * k1, axis=1).reshape(-1, k1, k1)
    f = np.where(backward_k_neigh_index == probe_indexes)
    nn_k1 = []
    for i in range(num_probes):
        fi = f[1][np.where(f[0] == i)[0]]
        nn_k1.append(forward_k_neigh_index[i, fi])
    return nn_k1


def compute_jaccard_distance(target_features, k1=20, k2=6, print_flag=False, search_option=0, use_float16=False):
    end = time.time()
    if print_flag:
        print('Computing jaccard distance...')

    N = target_features.size(0)
    mat_type = np.float16 if use_float16 else np.float32

    if search_option == 0:
        # GPU + PyTorch CUDA Tensors (1)
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        _, initial_rank = search_raw_array_pytorch(res, target_features, target_features, k1+1)
        initial_rank = initial_rank.cpu().numpy()
    elif search_option == 1:
        # GPU + PyTorch CUDA Tensors (2)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = search_index_pytorch(index, target_features, k1+1)
        res.syncDefaultStreamCurrentDevice()
        initial_rank = initial_rank.cpu().numpy()
    elif search_option == 2:
        # GPU
        ngpus = faiss.get_num_gpus()
        index = index_init_gpu(ngpus, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1+1)
    else:
        # CPU
        index = index_init_cpu(target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1+1)

    # initial_rank = initial_rank[:, 1:]      # delete self
    nn_k1 = k_reciprocal_neigh(initial_rank, k1+1)
    nn_k1_half = k_reciprocal_neigh(initial_rank, int(np.around(k1/2))+1)

    # distances = euler_cosine_distances(target_features, target_features)
    # distances = distances / distances.max()

    V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = [k_reciprocal_index]
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index).shape[0] > 2/3*candidate_k_reciprocal_index.shape[0]:
                k_reciprocal_expansion_index.append(candidate_k_reciprocal_index)
        k_reciprocal_expansion_index = np.concatenate(k_reciprocal_expansion_index)
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  # element-wise unique
        dist = 2.-2*torch.mm(target_features[i].unsqueeze(0).contiguous(), target_features[k_reciprocal_expansion_index].t())
        # dist = distances[i, k_reciprocal_expansion_index].unsqueeze(0)
        if use_float16:
            V[i, k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(mat_type)
            # V[i, k_reciprocal_expansion_index] = torch.exp(-5*dist).view(-1).cpu().numpy().astype(mat_type)
            # V[i, k_reciprocal_expansion_index] = F.softmax(-dist/dist.max(), dim=1).view(-1).cpu().numpy().astype(mat_type)
        else:
            V[i, k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()
            # V[i, k_reciprocal_expansion_index] = torch.exp(-5*dist).view(-1).cpu().numpy()
            # V[i, k_reciprocal_expansion_index] = F.softmax(-dist/dist.max(), dim=1).view(-1).cpu().numpy()

    del nn_k1, nn_k1_half

    if k2 != 1:
        # V_qe = np.zeros_like(V, dtype=mat_type)
        # for i in range(N):
        #     V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        # V = V_qe
        # del V_qe
        V = V[initial_rank[:, :k2], :].mean(axis=1)

    del initial_rank

    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:, i] != 0)[0])  # len(invIndex)=all_num

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=mat_type)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]]+np.minimum(V[i, indNonZero[j]],
                                                                             V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2-temp_min)

    del invIndex, V

    jaccard_dist[jaccard_dist < 0] = 0.0
    if print_flag:
        print("Jaccard distance computing time cost: {}".format(time.time()-end))

    return jaccard_dist


def _complex(X, alpha):
    return 1 / np.sqrt(2) * torch.complex(torch.cos(alpha * torch.pi * X), torch.sin(alpha * torch.pi * X))


def euler_cosine_distances(X, Y, alpha=1.9):
    X_ = _complex(X, alpha)
    Y_ = _complex(Y, alpha)
    D = torch.sum(X_ ** 2, dim=1).unsqueeze(1) + torch.sum(Y_ ** 2, dim=1) - 2 * torch.mm(X_, Y_.t())
    return torch.abs(D)


if __name__ == '__main__':
    import torch.nn.functional as F
    features = F.normalize(torch.rand(1000, 2048, dtype=torch.float32, device='cuda'))
    compute_jaccard_distance(features)
    pass
