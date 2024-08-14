# --coding:utf-8--
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances


def _complex(X, alpha):
    return 1 / np.sqrt(2) * np.complex64(np.cos(alpha * np.pi * X) + 1j * np.sin(alpha * np.pi * X))


def euler_cosine_distances(X, Y, alpha=1.9):
    X_ = _complex(X, alpha)
    Y_ = _complex(Y, alpha)
    D = np.sum(X_ ** 2, axis=1)[:, np.newaxis] + np.sum(Y_ ** 2, axis=1) - 2 * np.dot(X_, Y_.T)
    return np.abs(D)


__distances_factory = {
    'Euclidean': euclidean_distances,
    'Cosine': cosine_distances,
    'Euler Cosine': euler_cosine_distances
}


def get_distances(X, Y, dist, norm=True):
    distances = __distances_factory[dist](X, Y)
    if norm:
        if dist == 'Cosine':
            return distances / 2
        else:
            return distances / distances.max()
    else:
        return distances


if __name__ == '__main__':
    # a = np.random.rand(2000, 2048).astype(np.float32)
    # a /= np.linalg.norm(a, axis=1, keepdims=True)
    a = np.load('features.npy')
    D1 = cosine_distances(a, a)
    D2 = euler_cosine_distances(a, a)
    D3 = euclidean_distances(a, a)

    D2[D1 == 0] = 0
    pass
    min1 = D1[D1 != 0].min()
    min2 = D2[D2 != 0].min()
    min3 = D3[D3 != 0].min()
    max1 = D1.max()
    max2 = D2.max()
    max3 = D3.max()
    pass
