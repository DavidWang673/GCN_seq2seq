import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh


def split_data(data, days, n_obs, n_pred, day_slot=288, channel_in=1):
    n = data.shape[1]
    day_batch_size = day_slot - (n_obs + n_pred) + 1
    temp_data = []
    for i in range(days):
        day_batch_data = []
        for j in range(day_batch_size):
            day_batch_data.append(data[j:j + (n_obs + n_pred)])
        temp_data.append(day_batch_data)
    return np.array(temp_data).reshape((days * day_batch_size, n_obs + n_pred, n, channel_in))


def data_gen(path, n_obs=12, n_pred=6):
    n_train, n_val, n_test = 34, 5, 5
    #data = pd.read_csv("../data/PeMS-M/V_228.csv",header=None).values
    data = pd.read_csv(path, header=None).values
    # print(data.shape,data.mean(axis=0),data.std(axis=0))
    #data = preprocessing.scale(data)
    # print(data.shape,data.mean(axis=0),data.std(axis=0))
    train = split_data(data[0:n_train * 24 * 12, :], n_train, n_obs, n_pred)
    val = split_data(data[n_train * 24 * 12:(n_train + n_val)
                          * 24 * 12, :], n_val, n_obs, n_pred)
    test = split_data(data[(n_train + n_val) * 24 *
                           12:, :], n_test, n_obs, n_pred)

    return train, val, test


def scaled_adjacency(A, sigma2=0.1, e=0.5):
    n = A.shape[0]
    A = A / 10000
    A2, A_mask = A * A, np.ones([n, n]) - np.identity(n)
    return np.exp(-A2 / sigma2) * (np.exp(-A2 / sigma2) >= e) * A_mask


def normalize_adj(adj):

    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))

    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def chebyshev_polynomials(adj, k):

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (
        2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return np.concatenate([t.todense() for t in t_k], axis=1)
