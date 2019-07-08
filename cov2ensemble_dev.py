#!/usr/bin/env python

# call as: python cov2ensemble.py 

# =======================================
# Version 0.7
# 8 July, 2019
# https://patternizer.github.io
# michael.taylor AT reading DOT ac DOT uk
# =======================================

from  optparse import OptionParser
import numpy as np
import scipy.linalg as la
from scipy.special import erfinv
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from netCDF4 import Dataset
import xarray
import seaborn as sns; sns.set(style='darkgrid')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


def FPE(x0,x1):

    FPE = 100.*(1.-np.linalg.norm(x0)/np.linalg.norm(x1))

    return FPE

def pc_plot(score,coeff,labels=None):

    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, c = 'b', alpha=0.2)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center', fontsize=6)
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center', fontsize=6)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

    return

def calc_cov2u(N, Xave, Xcov):
    '''
    Routine to estimate uncertainty from a covariance matrix using Monte Carlo sampling from the underlying distribution. Code adapted from routine coded by Jonathan Mittaz: get_harm.py
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

def generate_10_single(n):

    #
    # Half width as random number always +ve
    #

    ntrial = n
    step_size = 1./ntrial
    random_numbers=[]

    #
    # +ve case
    #

    for i in range(ntrial):

        rno = i*step_size + step_size*np.random.random()
        random_numbers.append(np.sqrt(2.)*erfinv(rno))

    #
    # -ve case
    #

    for i in range(ntrial):

        rno = i*step_size + step_size*np.random.random()
        random_numbers.append(-np.sqrt(2.)*erfinv(rno))

    return random_numbers

def generate_10(n):

    rand1 = generate_10_single(n)
    rand2 = generate_10_single(n)
    rand3 = generate_10_single(n)

    dist1 = np.mean(rand1)**2 + (np.std(rand1)-1.)**2
    dist2 = np.mean(rand2)**2 + (np.std(rand2)-1.)**2
    dist3 = np.mean(rand3)**2 + (np.std(rand3)-1.)**2

    if dist1 < dist2 and dist1 < dist3:
        random_numbers = np.copy(rand1)
    elif dist2 < dist1 and dist2 < dist3:
        random_numbers = np.copy(rand2)
    elif dist3 < dist1 and dist3 < dist2:
        random_numbers = np.copy(rand3)

    #
    # Now randomise the numbers (mix them up)
    #

    in_index = np.arange(2*n).astype(dtype=np.int32)
    in_index_rand = np.random.random(size=(2*n))
    sort_index = np.argsort(in_index_rand)
    index = in_index[sort_index]
    random_numbers = random_numbers[index]

    return random_numbers

###################################################
# MAIN
###################################################

if __name__ == "__main__":

    ch = 37
    # OLD run:  (this one): software_tag=3e8c463 / job dir=job_avhxx_v6_EIV_10x_11
    # NEW run:            : software_tag=4d111a1 / job_dir=job_avhxx_v6_EIV_10x_11

    # Harmonisation runs: /gws/nopw/j04/fiduceo/Users/rquast/processing/harmonisation/

    # drwxr-x--- 1 rquast users     0 Jul  8 13:07 3.0-087ab71
    # drwxr-x--- 1 rquast users     0 Jun 11 09:50 3.0-276990a
    # drwxr-x--- 1 rquast users     0 Jun 17 08:00 3.0-3a7cca1
    # drwxr-x--- 1 rquast users     0 Jun 20 12:20 3.0-4d111a1
    # drwxr-x--- 1 rquast users     0 Jul  8 13:07 3.0-58ac69e
    # drwxr-x--- 1 rquast users     0 Jun 10 08:56 3.0-9df61fa
    # drwxr-x--- 1 rquast users     0 Jul  8 07:41 3.0-a7f0550
    # drwxr-x--- 1 rquast users     0 Jul  8 13:06 3.0-bdcc9e0
    # drwxr-x--- 1 rquast users     0 Jun 10 09:34 3.0-becb532
    # drwxr-x--- 1 rquast users     0 Jan 18 08:33 3.0-d4d8604
    # drwxr-x--- 1 rquast users     0 Jun 17 08:03 3.0-de3f5de
    # -rw-r----- 1 rquast users 11367 Jun 11 12:05 README_runs.md
    # lrwxrwxrwx 1 rquast users    11 Jun 17 14:48 run10 -> 3.0-a7f0550
    # lrwxrwxrwx 1 rquast users    11 Jul  3 10:31 run14 -> 3.0-58ac69e
    # lrwxrwxrwx 1 rquast users    11 Jul  5 14:46 run15 -> 3.0-087ab71
    # lrwxrwxrwx 1 rquast users    11 Jul  8 09:23 run16 -> 3.0-bdcc9e0
    # lrwxrwxrwx 1 rquast users    11 May 27 10:00 run5 -> 3.0-4d111a1
    # lrwxrwxrwx 1 rquast users    11 Jun  5 10:00 run6 -> 3.0-becb532
    # lrwxrwxrwx 1 rquast users    11 Jun  7 11:34 run7 -> 3.0-9df61fa
    # lrwxrwxrwx 1 rquast users    11 Jun 11 08:49 run8 -> 3.0-de3f5de
    # lrwxrwxrwx 1 rquast users    11 Jun 11 11:31 run9 -> 3.0-3a7cca1

    harm_file = 'FIDUCEO_Harmonisation_Data_' + str(ch) + '.nc'    
    ds = xarray.open_dataset(harm_file)
    Xave = np.array(ds.parameter)
    Xcov = np.array(ds.parameter_covariance_matrix)
    Xcor = np.array(ds.parameter_correlation_matrix)
    Xu = np.array(ds.parameter_uncertainty)

    #------------------------------------------------
    # OPTIONS
    #------------------------------------------------
    N = 10000 # for draw matrix from Xcov
    variance_threshold = 0.99 # 99% of total variance
    FLAG_X = 2      # 0=normal, 1=multinormal, 2=Xcov
    FLAG_method = 2 # 0=EVD,    1=SVD,         2=PCA
    #------------------------------------------------

    if FLAG_X == 0:
        X = np.empty(shape=(N,len(Xave)))
        for i in range(len(Xave)): 
            X[:,i] = (np.random.normal(0,1,N)).T
    elif FLAG_X == 1:
        X = np.random.multivariate_normal(Xave, Xcov, size=N)
    else:
        X = Xcov

    X_mean = np.mean(X, axis=0)
    X_sdev = np.std(X, axis=0)
    X_centered = X - X_mean
    X_scaled = X / X_sdev
    X_standardised = (X - X_mean) / X_sdev
    X_cor = np.corrcoef(X_centered.T)
    X_cov = np.dot(X_centered.T, X_centered) / X.shape[0] # = np.cov(X_centered.T)
    X_cov_det = np.linalg.det(X_cov) # should be ~ product of eigenvalues
    X_u = calc_cov2u(N, Xave, Xcov)
    X_cov_Xu = np.matmul(np.matmul(np.diag(Xu),np.array(Xcor)),np.diag(Xu))

    #
    # Calc eigenvalues & eigenvectors
    #

    if FLAG_method == 0:
        eigenvalues, eigenvectors = np.linalg.eig(X_cov) # or (unsorted) using Xcor
        # eigenvalues = np.diag(np.dot(eigenvectors.T, np.dot(X_cov, eigenvectors)))
        # X_cov = np.matmul(np.matmul(eigenvectors, np.diag(eigenvalues)), eigenvectors.T)
    elif FLAG_method == 1:
        U,S,V = np.linalg.svd(X_centered.T, full_matrices=True)  # or using Xcor
        eigenvalues, eigenvectors = np.sqrt(S), V # or U.T
    else:
        pca = PCA(n_components=X.shape[1])
        # pca = PCA(0.999)
        # X_transformed = pca.fit_transform(X_centered)
        # X_transformed = pca.fit_transform(X_scaled)
        X_transformed = pca.fit_transform(X_standardised)
        eigenvalues, eigenvectors = pca.explained_variance_, pca.components_.T

    eigenvalues_sum = eigenvalues.sum() # should be ~ n_components
    eigenvalues_norm = eigenvalues/eigenvalues_sum
    eigenvalues_cumsum = eigenvalues_norm.cumsum()
    eigenvalues_prod = eigenvalues.prod() # should be ~ det(Xcov)
    eigenvalues_rank = np.arange(1,len(eigenvalues)+1) # for plotting

    #
    # Calc nPC for explained variance > variance_threshold
    #

    nPC = np.where(eigenvalues_cumsum > variance_threshold)[0][0]

    #
    # Plot eigenvalue scree plot
    #

    fig,ax = plt.subplots()
    ax.plot(eigenvalues_rank, eigenvalues_norm, linestyle='-', marker='.', color='b')
    ax.plot(eigenvalues_rank[nPC], eigenvalues_norm[nPC], marker='o', color='k', mfc='none')
    ax.plot(eigenvalues_rank, eigenvalues_cumsum, linestyle='-', marker='.', color='r')
    ax.plot(eigenvalues_rank[nPC], eigenvalues_cumsum[nPC], marker='o', color='k', mfc='none')
    ax.set_xlabel('principal component')
    ax.set_ylabel('relative variance', color='b')
    plt.tight_layout()
    plt.savefig('eigenspectrum.png')
    plt.close('all')

    #-----------------------------------------------------------------
    # Check rotation vector
    #-----------------------------------------------------------------
    
    #
    # Construct the projection matrix W and project data onto 1st 2 PCs:
    #

    W = eigenvectors.dot(np.diag(np.sqrt(eigenvalues))).T
    Y_standardised = X_standardised.dot(W) # = X_transformed

    fig,ax = plt.subplots()
#    plt.plot(X_standardised[:,0],X_standardised[:,1],'k.',alpha=0.2, label='X: z-score)')
    plt.plot(Y_standardised[:,0],Y_standardised[:,1],'b.',alpha=0.2, label='Y(PC1,PC2)')
#    for v,e in zip(eigenvalues[0:2], eigenvectors.T[:,0:2]):
#        plt.plot([0, 3*np.sqrt(v)*e[0]], [0, 3*np.sqrt(v)*e[1]], 'r-', lw=2)
    plt.savefig('Y.png')
    plt.close('all')

    #
    # Calculate transformation matrix from eigen decomposition: check on W
    #

    R, S = eigenvectors, np.diag(np.sqrt(eigenvalues))
    W = R.dot(S).T

    # Transform data with inverse transformation matrix T^-1 # for de-whitening

    Z_standardised = Y_standardised.dot(np.linalg.inv(W))
    C = np.cov(Z_standardised.T)

    #-----------------------------------------------------------------
    # Perturb harmonisation coefficients with Jon's contrained sampling
    #-----------------------------------------------------------------

    #
    # Calc 2 main PCs
    #

    pca = PCA(n_components=2)
    pca.fit(Xcov)
    X_pca = pca.transform(X_standardised)

    fig,ax = plt.subplots()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.2)
    plt.axis('equal')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    fig.savefig('principal_components_12.png')
    plt.close('all')

    #
    # Create 10 deltas of Xave using (un)constrained random sampling
    #

    n = 5
    random_unconstrained = np.array(generate_10_single(n))
    random_constrained = np.array(generate_10(n))
    X_constrained = []
    X_unconstrained = []

    for i in range((2*n)):
        
        X_constrained.append(X_pca[:,0] * random_constrained[i]) 
        X_unconstrained.append(X_pca[:,0] * random_unconstrained[i])       

    X_constrained = np.array(X_constrained)
    X_unconstrained = np.array(X_unconstrained)

    sd1 = np.std(X_constrained,axis=0)   # wrong! (fix re discussion with Chris) 
    dXave = sd1 * pca.components_.T[:,0] # wrong! (fix re discussion with Chris)

    idx0 = np.arange(0,27,3)
    idx1 = np.arange(1,28,3)
    idx2 = np.arange(2,29,3)

    # proj_PC = np.matmul(X_constrained,np.matmul(U,np.diag(S)**2)) # if SVD
    proj_PC = np.matmul(X_constrained,W)

    proj_PC1 = proj_PC[:,0]
    proj_PC2 = proj_PC[:,1]

    fig,ax = plt.subplots()
    plt.plot(proj_PC1, proj_PC2,  'b.', alpha=0.2)  
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig('principal_components_12.png')
    plt.close('all')

    print('** end')





