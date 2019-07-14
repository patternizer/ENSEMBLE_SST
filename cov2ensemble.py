#!/usr/bin/env python

# call as: python cov2ensemble.py 
# include plot code: plot_cov2ensemble.py
  
# =======================================
# Version 0.10
# 14 July, 2019
# https://patternizer.github.io/
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

#------------------------------------------------------------------------------
import crs as crs            # constrained random sampling code: crs.py
import cov2u as cov2u        # estimation of uncertainty from covariance matrix
import convert_func          # measurement equations & L<-->BT conversion
#------------------------------------------------------------------------------

def FPE(x0,x1):

    FPE = 100.*(1.-np.linalg.norm(x0)/np.linalg.norm(x1))

    return FPE

def pc_plot(PC,E,labels=None):

    xs = PC[:,0]
    ys = PC[:,1]
    n = E.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, c = 'b', alpha=0.2)
    for i in range(n):
        plt.arrow(0, 0, E[i,0], E[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(E[i,0]* 1.15, E[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center', fontsize=6)
        else:
            plt.text(E[i,0]* 1.15, E[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center', fontsize=6)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

    return

def cov2ev(X,c):
    '''
    Eigenvalue decomposition of the covariance matrix and calculation of the number of PCs needed to be retained to account for relative fraction (c) of the variance
    '''

    eigenvalues,eigenvectors = np.linalg.eig(Xcov)
    eigenvalues_cumsum = (eigenvalues/eigenvalues.sum()).cumsum()
    nPC = np.where(eigenvalues_cumsum > c)[0][0]

    ev = {}
    ev['eigenvectors'] = eigenvectors
    ev['eigenvalues'] = eigenvalues
    ev['eigenvalues_sum'] = eigenvalues.sum()
    ev['eigenvalues_norm'] = eigenvalues/eigenvalues.sum()
    ev['eigenvalues_cumsum'] = eigenvalues_cumsum
    ev['eigenvalues_prod'] = eigenvalues.prod() # should be ~ det(Xcov)
    ev['eigenvalues_rank'] = np.arange(1,len(eigenvalues)+1) # for plotting
    ev['nPC'] = nPC

    return ev

def calc_dX(n,eigenvalues,eigenvectors):
    '''
    Create (2*n) deltas of Xave using (un)constrained random sampling
    '''

    random_unconstrained = np.sort(np.array(crs.generate_10_single(n)))
    random_constrained = np.sort(np.array(crs.generate_10(n)))
    dX0_constrained = []
    dX0_unconstrained = []
    dX1_constrained = []
    dX1_unconstrained = []
    for i in range((2*n)):        
        dX0_constrained.append(random_constrained[i] * np.sqrt(eigenvalues[0]) * eigenvectors[:,0])
        dX0_unconstrained.append(random_unconstrained[i] * np.sqrt(eigenvalues[0]) * eigenvectors[:,0])
        dX1_constrained.append(random_constrained[i] * np.sqrt(eigenvalues[1]) * eigenvectors[:,1])
        dX1_unconstrained.append(random_unconstrained[i] * np.sqrt(eigenvalues[1]) * eigenvectors[:,1])

    dX = {}
    dX['dX0_constrained'] = np.array(dX0_constrained)
    dX['dX0_unconstrained'] = np.array(dX0_unconstrained)
    dX['dX1_constrained'] = np.array(dX1_constrained)
    dX['dX1_unconstrained'] = np.array(dX1_unconstrained)

    return dX

###################################################
# MAIN
###################################################

if __name__ == "__main__":

    #------------------------------------------------
    # OPTIONS
    #------------------------------------------------
    ch = 12
    n = 5 # --> (2*n) = 10 = number of ensemble members
    c = 0.99 # variance_threshold
    N = 10000 # for draw matrix from Xcov
    FLAG_crs_test = False
    #------------------------------------------------

#    software_tag = '3e8c463' # job dir=job_avhxx_v6_EIV_10x_11 (old runs)
#    software_tag = '4d111a1' # job_dir=job_avhxx_v6_EIV_10x_11 (new runs)
    software_tag = 'v0.3Bet' # job_dir=job_avhxx_v6_EIV_10x_11 (new runs)
    plotstem = '_'+str(ch)+'_'+software_tag+'.png'

    harm_file = 'FIDUCEO_Harmonisation_Data_' + str(ch) + '.nc'    
    ds = xarray.open_dataset(harm_file)
    Xave = np.array(ds.parameter)
    Xcov = np.array(ds.parameter_covariance_matrix)
    Xcor = np.array(ds.parameter_correlation_matrix)
    Xu = np.array(ds.parameter_uncertainty)

    ev = cov2ev(Xcov,c)
    dX = calc_dX(n,ev['eigenvalues'],ev['eigenvectors'])
    
    # =======================================
    # INCLUDE PLOT CODE:
    exec(open('plot_cov2ensemble.py').read())
    # =======================================

    plot_eigenspectrum(ev)
    plot_pc_deltas(dX,Xu)
    plot_crs()

    #------------------------------------------------        
    print('** end')







