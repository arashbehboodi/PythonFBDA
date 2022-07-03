__author__ = "Arash Behboodi"
"""
This is written for Python 3.5.
The implementation is based on:
Alfonso S. Bandeira,
Ten Lectures and Forty-Two Open Problems in the Mathematics of Data Science
"""
# General Libraries to be included.
import numpy as np
from numpy import linalg as la


def diffusionmap(data_matrix, n, eps, t, k):
    """
    data_matrix: the data matrix where samples are put in columns of the matrix
    t: t indicates the number of iterations.
    k: indicates the target dimension for dimensionality reduction
    eps: is the parameter of Guassian kernel that is used.
    """
    # Finding the distance matrix"
    covariance_matrix = data_matrix@data_matrix.T

    distance_mat_1 = -2*covariance_matrix
    distance_mat_2 = np.outer(np.diag(covariance_matrix), np.ones((n, 1))) + np.outer(np.ones((n, 1)), np.diag(covariance_matrix).T)
    distance_mat = distance_mat_1+distance_mat_2
    # Kernel function for weight matrx: Gaussian Kernel
    # Constructing the weight matrix
    W = np.exp(-distance_mat/eps)
    # Degree matrix
    Deg = W@np.ones((n, 1))
    D = np.diag(Deg.reshape(n,))
    # Transition matrix
    M = la.inv(D)@W
    # Constructing the matrix S - obtaining diffusion vectors
    S = D**(1/2)@M@la.inv(D)**(1/2)
    # Spectral decomposition
    eigvalCov, eigvecCov = la.eig(S)
    idx = eigvalCov.argsort()[::-1]
    eigvalCov = eigvalCov[idx]
    eigvecCov = eigvecCov[:, idx]
    # Diffusion Map
    phiD = la.inv(D)**(1/2)@eigvecCov
    lambdaD = eigvalCov**t
    ##########################
    # Final Matrix with columns as the vectors
    DiffM = np.diag(lambdaD)@phiD.T
    Difftruncated = DiffM[1:k+1, :]
    return Difftruncated


def main():
    pass


if __name__ == 'main':
    main()
