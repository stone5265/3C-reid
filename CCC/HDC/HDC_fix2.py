import numpy as np
from daal4py.sklearn._utils import getFPType
from daal4py.sklearn.cluster._k_means_0_23 import _daal4py_compute_starting_centroids
try:
    # from ._HDC_utils import _relocate_empty_clusters, _harmonic_discrepancy
    from .distances import get_distances
    from .utils import _kmeans_plusplus as _kmeans_pp2
except:
    # from _HDC_utils import _relocate_empty_clusters, _harmonic_discrepancy
    from distances import get_distances
    from utils import _kmeans_plusplus as _kmeans_pp2

# VERSION = 'hd=norm_dist'
VERSION = 'hd=1-exp(-5*norm_dist)'
# VERSION = 'hd=2/(1+exp(-5*norm_dist))-1'
# VERSION = 'hd=norm_dist**2'
# VERSION = 'hd=norm_dist**3'
hd_formula_factory = {
    'hd=norm_dist': lambda x: x,
    'hd=1-exp(-5*norm_dist)': lambda x: 1-np.exp(-5*x),
    'hd=2/(1+exp(-5*norm_dist))-1': lambda x: 2/(1+np.exp(-5*x))-1,
    'hd=norm_dist**2': lambda x: x**2,
    'hd=norm_dist**3': lambda x: x**3,
}


def kmeans_plusplus(X, n_clusters, similarity='Euclidean'):
    X_fptype = getFPType(X)
    _, centroids = _daal4py_compute_starting_centroids(
        X, X_fptype, n_clusters, 'k-means++', 0, np.random.mtrand._rand)
    labels = get_distances(X, centroids, similarity).argmin(axis=1).astype(np.int32)

    return centroids, labels


def kmeans_plusplus2(X, n_clusters, xx_dist):
    centroids, indices = _kmeans_pp2(X, n_clusters, xx_dist, np.random.RandomState())
    labels = xx_dist[:, indices].argmin(axis=1).astype(np.int32)

    return centroids, labels


def random_init(X, n_clusters, normalize_centroid=False):
    labels = np.random.randint(0, n_clusters, X.shape[0], dtype=np.int32)
    centroids = _get_centroids(X, labels, n_clusters, normalize_centroid)

    return centroids, labels


class HDC(object):
    def __init__(
            self,
            n_clusters=8,
            max_iter=100,
            min_iter=10,
            init='k-means++',
            normalize_centroid=False,
            similarity='Euclidean',
            n_sigma=1.,
            threshold=None,
            min_instances=0,
            n_tol=4,
            logger=None,
            verbose=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.init = init
        self.normalize_centroid = normalize_centroid
        self.similarity = similarity
        self.n_sigma = n_sigma
        self.threshold = threshold
        self.min_instances = min_instances
        self.n_tol = n_tol
        self.logger = logger
        self.verbose = verbose

    def fit(self, X, mat_dist=None, outlier=True):
        n_clusters = self.n_clusters
        tol_count = 0
        shift_total_min = 0xfffff
        if self.logger is not None:
            self.logger.new_epoch()
        if mat_dist is None:
            mat_dist = get_distances(X, X, self.similarity)

        # init
        if self.init == 'k-means++':
            centroids, labels = kmeans_plusplus(X, n_clusters, self.similarity)
        elif self.init == 'k-means++2':
            centroids, labels = kmeans_plusplus2(X, n_clusters, mat_dist)
        else:
            centroids, labels = random_init(X, n_clusters, self.normalize_centroid)
        labels_pod = labels

        xx_hd_mat = _hd_mat(X, similarity=self.similarity, mat_dist=mat_dist)

        for iter in range(self.max_iter):
            if self.verbose:
                print(f' {iter}', end='')

            centroids_old = centroids.copy()
            labels, labels_pod, centroids = self.__iter(X, centroids, labels, xx_hd_mat)
            centroid_shift_total = ((centroids_old - centroids)**2).sum()
            self.__record(iter, centroid_shift_total)

            if centroid_shift_total == 0:
                break
            if iter >= self.min_iter:
                if centroid_shift_total < shift_total_min:
                    tol_count = 0
                    shift_total_min = centroid_shift_total
                elif tol_count < self.n_tol:
                    tol_count += 1
                else:
                    break
        if self.verbose:
            print()
        self.n_clusters = n_clusters
        self.cluster_centers_ = centroids
        self.labels_ = labels if not outlier else labels_pod

        return self

    def __iter(self,
               X,
               centroids,
               labels,
               xx_hd_mat,
               update_centroids=True):
        labels, labels_hd = _update_labels(X, centroids, labels, self.similarity, xx_hd_mat, self.verbose)
        labels_pod = labels
        if update_centroids:
            labels_pod = _peripheral_object_detection(labels, labels_hd, self.n_clusters, self.n_sigma, self.threshold)
            centroids = _get_centroids(X, labels_pod, self.n_clusters, self.normalize_centroid)
        return labels, labels_pod, centroids

    def __record(self, iter, centroid_shift):
        if self.logger is not None:
            self.logger.record(iter, centroid_shift)


def _get_centroids(X, labels, n_clusters, normalize=False):
    n_features = X.shape[1]

    centroids = np.zeros(shape=(n_clusters, n_features), dtype=np.float32)
    for k in range(n_clusters):
        centroids[k] = X[labels == k].mean(axis=0)

    if normalize:
        centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    return centroids


def _hd_mat(X, Y=None, similarity='Euclidean', mat_dist=None):
    if Y is None:
        Y = X

    if mat_dist is not None:
        hd_mat = 1 - np.exp(-mat_dist)
        # hd_mat = mat_dist
        # hd_mat = hd_formula_factory[VERSION](mat_dist)
    else:
        dist = get_distances(X, Y, similarity)
        hd_mat = hd_formula_factory[VERSION](dist)

    hd_mat[np.isnan(hd_mat)] = 1.
    hd_mat[hd_mat > 1] = 1.
    hd_mat[hd_mat < 0] = 0.

    return hd_mat


def _update_labels(X, centroids, labels, similarity='Euclidean', xx_hd_mat=None, verbose=0):
    n_samples = X.shape[0]
    n_clusters = centroids.shape[0]

    if xx_hd_mat is None:
        xx_hd_mat = _hd_mat(X, similarity=similarity)

    xc_mem_mat = 1 - _hd_mat(X, centroids, similarity=similarity)

    hd_mat = np.zeros((n_samples, n_clusters), dtype=np.float32)
    _harmonic_discrepancy(xx_hd_mat, xc_mem_mat, labels, hd_mat)

    # update labels
    # labels = hd_mat.argmin(axis=1).astype(np.int32)
    mem_mat = (hd_mat.sum(axis=1, keepdims=True) - hd_mat) / (n_clusters-1)
    labels = mem_mat.argmax(axis=1).astype(np.int32)
    labels_hd = hd_mat[np.arange(n_samples), labels]
    labels_mem = mem_mat[np.arange(n_samples), labels]

    flag = True
    num_repeat = 0
    while flag:
        count_in_clusters = np.zeros((n_clusters,), dtype=np.int32)
        _relocate_empty_clusters(hd_mat, labels_mem, count_in_clusters, labels, labels_hd)
        flag = False
        if not np.all(count_in_clusters > 0):
            num_repeat += 1
            flag = True
            if verbose:
                print('.', end='')

    return labels, labels_hd


def _peripheral_object_detection(labels, labels_hd, n_clusters, n_sigma=1., threshold_fixed=None):
    labels_pod = labels.copy()
    if threshold_fixed is None:
        for j in range(n_clusters):
            index = (labels == j)
            select_hd = labels_hd[index]
            if select_hd.shape[0] > 1:
                labels_hd_mean = np.mean(select_hd)
                labels_hd_std = select_hd.std(ddof=1)
            else:
                continue
            threshold = labels_hd_mean + n_sigma * labels_hd_std
            index &= (labels_hd > threshold)
            labels_pod[index] = -1
    else:
        labels_pod[labels_hd > threshold_fixed] = -1

    return labels_pod


if __name__ == '__main__':
    # test()
    features = np.load('test_avg/features.npy')
    rerank_dist = np.load('test_avg/rerank_dist.npy')
    # cluster = HDC(750, init='random', max_iter=30, min_instances=4)
    cluster = HDC(750, init='k-means++2')
    cluster.fit(features, mat_dist=rerank_dist)
    # print(cluster.n_clusters)
    # print(time()-start_time)
    #
    labels_index, labels_count = np.unique(cluster.labels_[cluster.labels_ != -1], return_counts=True)
    labels_count.sort()
    labels_count = labels_count[-1::-1]
    np.save('c0.npy', labels_count)
    pass
    # removed_labels = labels_index[np.where(labels_count < 4)]
    # print(removed_labels.size)
    # pass
    # print(cluster.labels_)