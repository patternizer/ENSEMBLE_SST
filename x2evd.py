#!/usr/bin/env python

# =======================================
# Version 0.7
# 8 July, 2019
# https://patternizer.github.io
# michael.taylor AT reading DOT ac DOT uk
# =======================================

import numpy as np
import scipy.linalg
import sklearn
from sklearn.decomposition import PCA

def x2evd(X,c):

    if X.shape[0] ~= X.shape[1]:

        if X.shape[0] < X.shape[1]:
            X = X.T
        
        X_mean = np.mean(X, axis=0)
        X_sdev = np.std(X, axis=0)
        X_centered = X - X_mean
        X_cov = np.cov(X_centered.T)
        U,S,V = np.linalg.svd(X_centered.T, full_matrices=True) 
        eigenvalues, eigenvectors = np.sqrt(S), V # or U.T

    else:

        pca = PCA(n_components=X.shape[1])
        X_transformed = pca.fit_transform(X)
        eigenvalues, eigenvectors = pca.explained_variance_, pca.components_.T

    eigenvalues_cumsum = (eigenvalues/eigenvalues.sum()).cumsum()
    nPC = np.where(eigenvalues_cumsum > c)[0][0]

    return nPC, eigenvalues, eigenvectors


