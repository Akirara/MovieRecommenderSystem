import numpy as np


def initialize(R, K):
    """
    Return initial matrices for an N * M matrix R and K features

    :param R: the matrix to be factorized
    :param K: the number of latent features
    :return: P, Q initial matrices of N * K and M * K sizes
    """
    N, M = R.shape
    P = np.random.rand(N, K)
    Q = np.random.rand(M, K)

    return P, Q


def factor(R, P=None, Q=None, K=2, steps=5000, alpha=0.0002, beta=0.02):
    """
    Perform matrix factorization on R with given parameters

    :param R: an N * M matrix to be factorized
    :param P: an initial N * K matrix
    :param Q: an initial M * K matrix
    :param K: the number of latent features
    :param steps: the maximum number of iterations
    :param alpha: the learning rate
    :param beta: the regularization parameter
    :return: final matrices P & Q
    """
    if not P or not Q:
        P, Q = initialize(R, K)
    Q = Q.T

    rows, cols = R.shape
    for step in range(steps):
        for i in range(rows):
            for j in range(cols):
                if R[i, j] > 0:
                    eij = R[i, j] - np.dot(P[i, :], Q[:, j])
                    for k in range(K):
                        P[i, k] = P[i, k] + alpha * (2 * eij * Q[k, j] - beta * P[i, k])
                        Q[k, j] = Q[k, j] + alpha * (2 * eij * P[i, k] - beta * Q[k, j])
        e = 0
        for i in range(rows):
            for j in range(cols):
                if R[i, j] > 0:
                    e += pow(R[i, j] - np.dot(P[i, :], Q[:, j]), 2)
                    for k in range(K):
                        e += (beta / 2) * (pow(P[i, k], 2) + pow(Q[k, j], 2))
        if e < 0.001:
            break

    return P, Q.T
