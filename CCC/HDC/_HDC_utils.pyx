import numpy as np
from cython cimport floating
from cython.parallel cimport prange
from cython import boundscheck, wraparound

# Speed up Version 24/03/05

@boundscheck(False)
@wraparound(False)
cpdef void _harmonic_discrepancy(
        floating[:, :] xx_hd_mat,   # IN
        floating[:, :] xc_mem_mat,  # IN
        int[:] labels,              # IN
        floating[:, :] hd_mat       # INOUT
):
    cdef:
        long n_samples = xx_hd_mat.shape[0]
        long n_clusters = xc_mem_mat.shape[1]
        long i, j, k
        int[:, :] max_k = np.zeros((n_samples, n_clusters), dtype=np.int32)
        floating[:, :] max_harmonic_average = -np.ones((n_samples, n_clusters), dtype=np.float32)
        floating[:] tmp_harmonic_average = np.zeros((n_samples,), dtype=np.float32)

    for i in prange(n_samples, nogil=True):
        for k in range(n_samples):
            if labels[k] != -1:
                tmp_harmonic_average[i] = xx_hd_mat[i, k] * xc_mem_mat[k, labels[k]]\
                                          / (xx_hd_mat[i, k] + xc_mem_mat[k, labels[k]])
                if tmp_harmonic_average[i] > max_harmonic_average[i, labels[k]]:
                    max_harmonic_average[i, labels[k]] = tmp_harmonic_average[i]
                    max_k[i, labels[k]] = k

    for i in prange(n_samples, nogil=True):
        for j in range(n_clusters):
            hd_mat[i, j] = xx_hd_mat[i, max_k[i, j]]


cpdef void _relocate_empty_clusters(
        floating[:, :] hd_mat,       # IN
        floating[:] labels_mem,   # IN
        int[:] count_in_clusters,    # INOUT
        int[:] labels,               # INOUT
        floating[:] labels_hd        # INOUT
):
    """Relocate centers which have no sample assigned to them."""
    cdef:
        int n_samples = hd_mat.shape[0]
        int n_clusters = hd_mat.shape[1]
        int i

    for i in range(n_samples):
        count_in_clusters[labels[i]] += 1

    cdef:
        int[:] empty_clusters = np.where(np.equal(count_in_clusters, 0))[0].astype(np.int32)
        int n_empty = empty_clusters.shape[0]

    if n_empty == 0:
        return

    cdef:
        floating[:] memberships = np.copy(labels_mem)
        int[:] mini_clusters = np.where(np.less(count_in_clusters, 5))[0].astype(np.int32)
        int[:] indexes
        int j

    for i in range(mini_clusters.shape[0]):
        indexes = np.where(np.equal(labels, mini_clusters[i]))[0].astype(np.int32)
        for j in range(indexes.shape[0]):
            memberships[indexes[j]] = 1.

    cdef:
        #int[:] far_from_centers = np.argpartition(distances, -n_empty)[:-n_empty-1:-1].astype(np.int32)
        int[:] far_from_centers = np.argpartition(memberships, n_empty)[:n_empty].astype(np.int32)
        int new_cluster_id, far_idx, offset=0

    for i in range(n_empty):

        new_cluster_id = empty_clusters[i]
        far_idx = far_from_centers[i]
        #far_idx = far_from_centers[i + offset]

        #while count_in_clusters[labels[far_idx]] < 2:
        #    offset += 1
        #    far_idx = far_from_centers[i + offset]

        count_in_clusters[labels[far_idx]] -= 1
        count_in_clusters[new_cluster_id] += 1
        labels[far_idx] = new_cluster_id
        labels_hd[far_idx] = hd_mat[far_idx, new_cluster_id]
