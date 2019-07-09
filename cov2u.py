import numpy as np
import scipy.linalg

def calc_cov2u(N, Xave, Xcov):
    '''
    Routine to estimate uncertainty from a covariance matrix using Monte Carlo sampling of the normal distribution and projection.  the underlying distribution. Adapted from get_harm.py written by Jonathan Mittaz.

    # =======================================
    # Version 0.1
    # 9 July, 2019
    # https://patternizer.github.io/
    # michael.taylor AT reading DOT ac DOT uk
    # =======================================

    '''

    eigenval, eigenvec = np.linalg.eig(Xcov)
    T = np.matmul(eigenvec, np.diag(np.sqrt(eigenval)))
    ndims = Xcov.shape[1]
    position = np.zeros((N, ndims))
    draws = np.zeros((N, ndims))
    for j in range(ndims):
        position[:,:] = 0.
        position[:,j] = np.random.normal(size=N, loc=0., scale=1.)
        for i in range(position.shape[0]):
            vector = position[i,:]
            ovector = np.matmul(T,vector)
            draws[i,:] = draws[i,:]+ovector

    Xu = np.std(draws, axis=0)

    return Xu
