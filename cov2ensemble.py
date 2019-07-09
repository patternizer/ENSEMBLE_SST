#!/usr/bin/env python

# call as: python cov2ensemble.py 
  
# =======================================
# Version 0.8
# 9 July, 2019
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

import crs as crs            # constrained random sampling code: crs.py
import cov2u as cov2u        # estimation of uncertainty from covariance matrix
import convert_func          # measurement equations & L<-->BT conversion

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
    ch = 37
    n = 5 # --> (2*n) = 10 = number of ensemble members
    c = 0.99 # variance_threshold
    N = 10000 # for draw matrix from Xcov
    FLAG_crs_test = False
    #------------------------------------------------

#    software_tag = '3e8c463' # job dir=job_avhxx_v6_EIV_10x_11 (old runs)
#    software_tag = '4d111a1' # job_dir=job_avhxx_v6_EIV_10x_11 (new runs)
#    software_tag = 'v0.3Bet' # job_dir=job_avhxx_v6_EIV_10x_11 (new runs)
    software_tag = 'v0.3Bet_all' # job_dir=job_avhxx_v6_EIV_10x_11 (new runs)
    plotstem = '_'+str(ch)+'_'+software_tag+'.png'

#    harm_file = 'FIDUCEO_Harmonisation_Data_' + str(ch) + '.nc'    
    harm_file = 'FIDUCEO_Harmonisation_Data_' + str(ch) + '_all' + '.nc'    
    ds = xarray.open_dataset(harm_file)
    Xave = np.array(ds.parameter)
    Xcov = np.array(ds.parameter_covariance_matrix)
    Xcor = np.array(ds.parameter_correlation_matrix)
    Xu = np.array(ds.parameter_uncertainty)

    ev = cov2ev(Xcov,c)
    dX = calc_dX(n,ev['eigenvalues'],ev['eigenvectors'])

    #------------------------------------------------
    # PLOTS
    #------------------------------------------------

    nPC = ev['nPC']
    fig,ax = plt.subplots()
    plt.plot(ev['eigenvalues_rank'], ev['eigenvalues_norm'], linestyle='-', marker='.', color='b', label=r'$\lambda/\sum\lambda$')
    plt.plot(ev['eigenvalues_rank'][nPC], ev['eigenvalues_norm'][nPC], marker='o', color='k', mfc='none',label=None)
    plt.plot(ev['eigenvalues_rank'], ev['eigenvalues_cumsum'], linestyle='-', marker='.', color='r',label='cumulative')
    plt.plot(ev['eigenvalues_rank'][nPC], ev['eigenvalues_cumsum'][nPC], marker='o', color='k', mfc='none',label='n(PC)='+str(nPC))
    plt.legend(loc='right', fontsize=10)
    plt.xlabel('rank')
    plt.ylabel('relative variance', color='b')
    plotstr = 'eigenspectrum'+plotstem
    plt.savefig(plotstr)
    plt.close('all')

    fig,ax = plt.subplots()
    for i in range(2*n):
        labelstr_c = 'PC1 (constrained) ' + str(i+1)
        plt.plot(dX['dX0_constrained'][i,:]/Xu, lw=2, label=labelstr_c)
    for i in range(2*n):
        labelstr_u = '(unconstrained) ' + str(i+1)
        plt.plot(dX['dX0_unconstrained'][i,:]/Xu, '.', label=labelstr_u)
#    plt.ylim(-2,2)
    plt.legend(loc=2, fontsize=6, ncol=4)
    plt.xlabel('parameter, a(n)')
    plt.ylabel(r'$\delta a(n)/u(n)$')
    plt.tight_layout()
    plotstr = 'pc1_deltas_over_Xu'+plotstem
    plt.savefig(plotstr)
    plt.close('all')

    fig,ax = plt.subplots()
    for i in range(2*n):
        labelstr_c = 'PC2 (constrained) ' + str(i+1)
        plt.plot(dX['dX1_constrained'][i,:]/Xu, lw=2, label=labelstr_c)
    for i in range(2*n):
        labelstr_u = '(unconstrained) ' + str(i+1)
        plt.plot(dX['dX1_unconstrained'][i,:]/Xu, '.', label=labelstr_u)
#    plt.ylim(-2,2)
    plt.legend(loc=2, fontsize=6, ncol=4)
    plt.xlabel('parameter, a(n)')
    plt.ylabel(r'$\delta a(n)/u(n)$')
    plt.tight_layout()
    plotstr = 'pc2_deltas_over_Xu'+plotstem
    plt.savefig(plotstr)
    plt.close('all')

    if FLAG_crs_test:

        for n in np.array([10,50,100,500,1000,5000,10000,50000]):
            
            random_numbers_unconstrained = crs.generate_10_single(n)
            random_numbers_constrained = crs.generate_10(n)

            fig,ax = plt.subplots(1,2)
            labelstr_constrained = 'n=' + str(n) + ': constrained'
            labelstr_unconstrained = 'n=' + str(n) + ': unconstrained'
            ax[0].plot(np.sort(np.array(random_numbers_unconstrained)), label=labelstr_unconstrained)
            ax[0].plot(np.sort(np.array(random_numbers_constrained)), label=labelstr_constrained)
            ax[0].legend(loc=2, fontsize=8)
            ax[0].set_ylim(-5,5)
            ax[0].set_ylabel('z-score')
            ax[0].set_xlabel('rank')
            ax[0].set_title(r'Sorted random sample from $erf^{-1}(x)$')
            ax[1].hist(random_numbers_unconstrained,bins=100,alpha=0.3,label='unconstrained')
            ax[1].hist(random_numbers_constrained,bins=100,alpha=0.3,label='constrained')
            ax[1].set_xlim(-5,5)
            ax[1].set_xlabel('z-score')
            ax[1].set_ylabel('count')
            plotstr = 'random_numbers_n_' + str(n) + '.png'
            plt.tight_layout()
            plt.savefig(plotstr)
            plt.close('all')

    print('** end')





