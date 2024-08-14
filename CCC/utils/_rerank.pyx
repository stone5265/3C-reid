# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

import numpy as np

from cython cimport floating
from cython.parallel cimport prange
from cython import boundscheck


cpdef void _V(
        floating[:, :] V,                 # INOUT
        floating[:, :] original_dist,     # IN
        int[:, :] initial_rank,           # IN
        const int k1,                     # IN
        const int all_num                 # IN
):
    cdef:
        int k1_half = int(np.around(k1/2.))
        # forward_k_neigh_index
        int[:] forward_index = np.zeros((k1+1,), dtype=np.int32)
        # backward_k_neigh_index
        int[:, :] backward_index = np.zeros((k1+1, k1+1), dtype=np.int32)
        # candidate_forward_k_neigh_index
        int[:] candidate_forward_index = np.zeros((k1_half+1,), dtype=np.int32)
        # candidate_backward_k_neigh_index
        int[:, :] candidate_backward_index = np.zeros((k1_half+1, k1_half+1), dtype=np.int32)

        # k_reciprocal_index
        int[:] index = np.zeros((k1+1,), dtype=np.int32)
        # k_reciprocal_expansion_index
        int[:] expansion_index
        # candidate_k_reciprocal_index
        int[:] candidate_index = np.zeros((k1_half+1,), dtype=np.int32)

        int len_index, len_expansion_index, len_candidate_index
        int i, j, k, candidate
        int[:] fi, fi_candidate
        floating sum_weight
        floating[:] weight = np.zeros((all_num,), dtype=np.float32)

    for i in range(all_num):
        # k-reciprocal neighbors
        for k in range(k1+1):
            forward_index[k] = initial_rank[i, k]
            backward_index[k, :] = initial_rank[forward_index[k], :k1+1]
        fi = np.where(np.equal(backward_index, i))[0].astype(np.int32)
        len_index = 0
        for k in range(len(fi)):
            index[k] = forward_index[fi[k]]
            len_index += 1
        expansion_index = index[:len_index]

        for j in range(len_index):
            candidate = index[j]
            for k in range(k1_half+1):
                candidate_forward_index[k] = initial_rank[candidate, k]
                candidate_backward_index[k, :] = initial_rank[candidate_forward_index[k], :k1_half+1]
            fi_candidate = np.where(np.equal(candidate_backward_index, candidate))[0].astype(np.int32)
            len_candidate_index = 0
            for k in range(len(fi_candidate)):
                candidate_index[k] = candidate_forward_index[fi_candidate[k]]
                len_candidate_index += 1
            if len(np.intersect1d(candidate_index[:len_candidate_index], index[:len_index])) \
                    > 2./3*len_candidate_index:
                expansion_index = np.append(expansion_index, candidate_index[:len_candidate_index])

        expansion_index = np.unique(expansion_index)

        len_expansion_index = len(expansion_index)
        sum_weight = 0.

        for k in range(len_expansion_index):
            weight[k] = np.exp(-original_dist[i, expansion_index[k]])
            sum_weight += weight[k]
        for k in range(len_expansion_index):
            V[i, expansion_index[k]] = weight[k] / sum_weight


cpdef void _V_qe(
        floating[:, :] V_qe,
        floating[:, :] V,
        int[:, :] initial_rank,
        const int k2,
        const int all_num
):
    cdef:
        floating[:] temp
        int i, j, k

    with boundscheck(False):
        for i in prange(all_num, nogil=True):
            # V_qe[i, :] = np.mean(np.asarray(V)[initial_rank[i, :k2], :], axis=0)
            for j in range(k2):
                for k in range(all_num):
                    V_qe[i, k] += V[initial_rank[i, j], k] / k2


cpdef void _jaccard_dist(
        floating[:, :] jaccard_dist,
        floating[:, :] V,
        const int query_num,
        const int gallery_num
):
    cdef:
        int i, j, k
        floating[:] temp_min
        int[:, :] invIndex = np.zeros((gallery_num, gallery_num), dtype=np.int32)
        int[:] len_invIndex = np.zeros((gallery_num,), dtype=np.int32)

    with boundscheck(False):
        for i in prange(gallery_num, nogil=True):
            for j in range(gallery_num):
                if V[j, i] != 0:
                    invIndex[i, len_invIndex[i]] = j
                    len_invIndex[i] += 1

        for i in range(query_num):
            temp_min = np.zeros((gallery_num,), dtype=np.float32)
            for j in prange(gallery_num, nogil=True):
                if V[i, j] != 0:
                    for k in range(len_invIndex[j]):
                        temp_min[invIndex[j][k]] += min(V[i, j], V[invIndex[j][k], j])
            for j in prange(gallery_num, nogil=True):
                jaccard_dist[i, j] = 1 - temp_min[j]/(2.-temp_min[j])
