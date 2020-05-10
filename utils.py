#!/usr/bin/env python3

import scipy.sparse as sp
import numpy as np
import numpy.linalg as LA
from sklearn import preprocessing
from keras.utils import to_categorical
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence
import scipy.stats

VERBOSE = True

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


#     # build graph
#     import networkx as nx
#     g=nx.read_edgelist("{}{}.cites".format(path, dataset))
#     N=len(g)
#     adj=nx.to_numpy_array(g,nodelist=idx_features_labels[:, 0])
#     adj = sp.coo_matrix(adj)


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj += sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj


# def sample_mask(idx, l):
#     mask = np.zeros(l)
#     mask[idx] = 1
#     return np.array(mask, dtype=np.bool)

def random_mask(p, l):
    b = scipy.stats.bernoulli(p)
    return b.rvs(l).astype(np.bool)


# def get_splits(y):
#     idx_train = range(140)
#     idx_val = range(200, 500)
#     idx_test = range(500, 1500)
    
#     y_train = np.zeros(y.shape, dtype=np.int32) #y:label
#     y_val = np.zeros(y.shape, dtype=np.int32)
#     y_test = np.zeros(y.shape, dtype=np.int32)

#     y_train[idx_train] = y[idx_train]
#     y_val[idx_val] = y[idx_val]
#     y_test[idx_test] = y[idx_test]

#     train_mask = sample_mask(idx_train, y.shape[0])
#     val_mask = sample_mask(idx_val, y.shape[0])
#     test_mask = sample_mask(idx_test, y.shape[0])

#     return y_train, y_val, y_test, train_mask,val_mask,test_mask


def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian


def rescale_laplacian(laplacian):
    try:
        if VERBOSE:
            print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        if VERBOSE:
            print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian


def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    if VERBOSE:
        print("Calculating Chebyshev polynomials up to order {}...".format(k))

    T_k = [sp.eye(X.shape[0]).tocsr(), X]

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k+1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k


# extended by William
def load_from_csv(train_path="train.csv", test_path="test.csv", normalized=False, *args, **kwargs):
    X_y = np.genfromtxt(train_path, delimiter=',', *args, **kwargs)
    X_train = X_y[:, :-1]
    y_train = encode_onehot(X_y[:, -1])
    X_y = np.genfromtxt(test_path, delimiter=',', *args, **kwargs)
    X_test = X_y[:, :-1]
    y_test = encode_onehot(X_y[:, -1])
    if normalized:
        X_train /= X_train.sum(1).reshape(-1, 1)
        X_test /= X_test.sum(1).reshape(-1, 1)
    return X_train, X_test, y_train, y_test


# similarity: (x1, x2) |-> [0,1]
# x1==x2 -> 1
def cosine(x1, x2):
    return np.dot(x1, x2)/LA.norm(x1)*LA.norm(x2)

def hamming(x1, x2):
    return np.mean(x1==x2)

def norm(x1, x2, p=2):
    # 1/(||x1-x2||+1)
    return 1 / (LA.norm(x1-x2, p)+1)

def make_adj(X_train, X_test, similarity=cosine):
    X_total = np.vstack((X_train, X_test))
    edges = []
    N = X_total.shape[0]
    for i in range(N):
        for j in range(i+1, N):
            if similarity(X_total[i,:], X_total[j,:]) > 0.75:
                edges.append([i,j])
    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(N, N), dtype=np.float32)

    # build symmetric adjacency matrix
    adj += adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj
