import numpy as np

# from time import time
from ._rerank import _V, _V_qe, _jaccard_dist


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):

    # start_time = time()##################
    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    # print('[0.1]use {}s.'.format(time()-start_time))##################

    # start_time = time()##################
    original_dist = np.power(original_dist, 2).astype(np.float32)
    # print('[0.2]use {}s.'.format(time()-start_time))##################

    # start_time = time()##################
    original_dist = np.transpose(1. * original_dist/np.max(original_dist, axis=0)).astype(np.float32, order='C')
    # print('[0.3]use {}s.'.format(time() - start_time))  ##################

    # start_time = time()  ##################
    initial_rank = np.argsort(original_dist).astype(np.int32)
    # print('[0.4]use {}s.'.format(time()-start_time))##################

    V = np.zeros_like(original_dist).astype(np.float32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    # start_time = time()##################

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1+1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2.))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index, :int(np.around(k1/2.))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index)) > 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)

    # print('[1]use {}s.'.format(time() - start_time))##################

    original_dist = original_dist[:query_num]

    # start_time = time()##################
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    # print('[2]use {}s.'.format(time() - start_time))##################

    # start_time = time()##################
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + \
                                        np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min/(2.-temp_min)
    # print('[3]use {}s.'.format(time() - start_time))##################

    final_dist = lambda_value*original_dist + (1-lambda_value)*jaccard_dist
    return final_dist[:query_num, query_num:]


def re_ranking2(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist, axis=0)).astype(np.float32, order='C')

    initial_rank = np.argpartition(original_dist, np.arange(max(k1+1, k2))).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    V = cal_V(original_dist, initial_rank, k1, all_num)

    original_dist = original_dist[:query_num]

    if k2 != 1:
        V = cal_V_qe(V, initial_rank, k2, all_num)

    jaccard_dist = cal_jaccard_dist(V, query_num, gallery_num)

    final_dist = lambda_value*original_dist + (1-lambda_value)*jaccard_dist
    return final_dist[:query_num, query_num:]


def cal_V(original_dist, initial_rank, k1, all_num):
    V = np.zeros_like(original_dist).astype(np.float32, order='C')
    _V(V, original_dist, initial_rank, k1, all_num)
    return V


def cal_V_qe(V, initial_rank, k2, all_num):
    V_qe = np.zeros_like(V, dtype=np.float32, order='C')
    _V_qe(V_qe, V, initial_rank, k2, all_num)
    return V_qe


def cal_jaccard_dist(V, query_num, gallery_num):
    jaccard_dist = np.zeros((query_num, gallery_num), dtype=np.float32, order='C')
    _jaccard_dist(jaccard_dist, V, query_num, gallery_num)
    return jaccard_dist


if __name__ == '__main__':
    qg = np.ones((10, 100))
    qq = np.ones((10, 10))
    gg = np.ones((100, 100))
    re_ranking2(qg, qq, gg)
    from time import time
    q_g_dist = np.random.rand(2000, 10000)
    q_q_dist = np.random.rand(2000, 2000)
    g_g_dist = np.random.rand(10000, 10000)

    start_time = time()
    result1 = re_ranking(q_g_dist, q_q_dist, g_g_dist)
    print('use {}s.'.format(time()-start_time))

    print()

    # start_time = time()
    # rerank = ReRank(n_jobs=2)
    # rerank.fit(q_g_dist, q_q_dist, g_g_dist)
    # print('use {}s.'.format(time()-start_time))

    start_time = time()
    result2 = re_ranking2(q_g_dist, q_q_dist, g_g_dist)
    print('use {}s.'.format(time()-start_time))

    print(np.allclose(result1, result2))
