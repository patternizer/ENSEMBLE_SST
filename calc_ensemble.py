#!/usr/bin/env python

# ipdb> import os; os._exit(1)

# call as: python calc_ensemble.py

# =======================================
# Version 0.3
# 26 March, 2019
# michael.taylor AT reading DOT ac DOT uk
# =======================================

import os  
import os.path  
import glob  
import optparse 
from  optparse import OptionParser  
import sys   
import numpy as np
import numpy.ma as ma  
import xarray
import pandas as pd
from pandas import Series, DataFrame, Panel
from sklearn.preprocessing import StandardScaler
#import arpack
import scipy
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
import seaborn as sns; sns.set(style="darkgrid")
import matplotlib.pyplot as plt; plt.close("all")
import matplotlib.dates as mdates

# =======================================    

def load_data(file_in):
    '''
    Load harmonisation parameters and covariance matrix
    '''
    # matchup_dataset = "AVHRR_REAL_4_RSA_____"
    # job = "job_avh11_v6_ODR_101_11.nml"
    # matchup_dataset_begin = "19911004"
    # matchup_dataset_end = "20151231"

    ds = xarray.open_dataset(file_in)

    return ds

def calc_eigen(ds):
    '''
    Calculate eigenvalues and eigenvectors from the harmonisation parameter covariance matrix
    '''
    # Harmonisation parameter covariance matrix: (27, 27)
    parameter_covariance_matrix = ds['parameter_covariance_matrix'] 
    X = parameter_covariance_matrix

    eigenval, eigenvec = np.linalg.eig(X)
#    print('Eigenvalues \n%s' %eigenval)   
#    print('Eigenvectors \n%s' %eigenvec)

    return eigenval, eigenvec

def plot_eigenval(eigenval):
    '''
    Plot eigenvalues as a scree plot
    '''
    Y = eigenval / max(eigenval)
    N = len(Y)    
    X = np.arange(0,N)

    fig = plt.figure()
    plt.fill_between(X, Y, step="post", alpha=0.4)
    plt.plot(X, Y, drawstyle='steps-post')
    plt.tick_params(labelsize=12)
    plt.ylabel("Relative value", fontsize=12)
    title_str = 'Scree plot: eigenvalue max=' + "{0:.5f}".format(eigenval.max())
    plt.title(title_str)
    plt.savefig('eigenvalues.png')    

def plot_eigenvec(eigenvec):
    '''
    Plot eigenvector matrix as a heatmap
    '''
    X = eigenvec

    fig = plt.figure()
    sns.heatmap(X, center=0, linewidths=.5, cmap="viridis", cbar=True)
#    sns.heatmap(X, vmin=0, vmax=1, linewidths=.5, cmap="viridis", cbar=True)
#    sns.heatmap(X, center=0, linewidths=.5, annot=True, fmt="f", cmap="viridis", cbar=True)

    #
    # Mask out upper triangle
    #

    # mask = np.zeros_like(X)
    # mask[np.triu_indices_from(mask)] = True
    # with sns.axes_style("white"):
    #    sns.heatmap(X, mask=mask, square=True)

    plt.title('Eigenvector matrix')
    plt.savefig('eigenvectors.png')    

def plot_covariance(ds):
    '''
    Plot harmonisation parameter covariance matrix as a heatmap
    '''
    # Harmonisation parameter covariance matrix: (27, 27)
    parameter_covariance_matrix = ds['parameter_covariance_matrix'] 
    X = parameter_covariance_matrix

    fig = plt.figure()
    sns.heatmap(X, center=0, linewidths=.5, cmap="viridis", cbar=True)
    plt.title('Covariance matrix')
    plt.savefig('covariance_matrix.png')    

def plot_sample_binormal():
    '''
    # -------------------------------
    # TEST CASE: SAMPLE FROM BINORMAL
    # -------------------------------
    '''

    #
    # Generate random binormal data
    #
    Xmean = np.zeros(2) # [0,0]
    Xcov = np.eye(2) # [[1,0],[0,1]]
    size = 10000
    data1 = np.random.multivariate_normal(Xmean, Xcov, size)

    #
    # Make 100 draws 
    #

#    X1 = np.random.rand(100)
#    X2 = np.random.rand(100)
    X1 = data1[:,0]
    X2 = data1[:,1]
    Xmean = [X1.mean(), X2.mean()]
    X = np.stack((X1, X2), axis=0)
    Xcov = np.cov(X)
#    Xcov = [[0.9,0.1],[0.1,0.9]]
    print(Xmean)
    print(Xcov)
    size = 100
    data2 = np.random.multivariate_normal(Xmean, Xcov, size)        
    df1 = pd.DataFrame(data1, columns=['x1', 'x2'])
    df2 = pd.DataFrame(data2, columns=['y1', 'y2'])

    #
    # Plot joint distribution
    #

    # colours = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    #             purple,      blue,      grey,    orange,      navy,     green
    fig, ax = plt.subplots()
    graph = sns.jointplot(x=df1.x1, y=df1.x2, kind="hex", space=0, color="#9b59b6")
    plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
    cax = graph.fig.add_axes([.81, .15, .02, .5])  # x, y, width, height
    cbar = plt.colorbar(cax=cax)
    cbar.set_label('count')
    graph.x = df2.y1
    graph.y = df2.y2
    graph.plot_joint(plt.scatter, marker="x", color="#34495e", s=2)
    graph.x = X1.mean()
    graph.y = X2.mean()
    graph.plot_joint(plt.scatter, marker="x", color="#3498db", s=50)    
    fig.suptitle('2D-sampling from binormal distribution')
    plt.savefig('sampled_binormal.png')    

def calc_ensemble(ds):
    '''
    Sample from the N-normal distribution using the harmonisation parameters as the mean values (best case) and the covariance matrix as the N-variance
    '''
    # Harmonisation parameters: (27,)
    parameter = ds['parameter'] 
    # Harmonisation parameter uncertainties: (27,)
    parameter_uncertainty = ds['parameter_uncertainty'] 
    # Harmonisation parameter covariance matrix: (27, 27)
    parameter_covariance_matrix = ds['parameter_covariance_matrix'] 
    # Harmonisation parameter correlation matrix: (27, 27)
    parameter_correlation_matrix = ds['parameter_correlation_matrix'] 
    # Harmonisation parameter Hessian matrix (internal): (27, 27)
    parameter_hessian_matrix = ds['parameter_hessian_matrix'] 
    # Harmonisation parameter add offsets (internal): (27,)
    parameter_add_offset = ds['parameter_add_offset'] 
    # Harmonisation parameter scale factors (internal): (27,)
    parameter_scale_factor = ds['parameter_scale_factor'] 
    # Sensors associated with harmonisation parameters: (27,)
    parameter_sensors = ds['parameter_sensors'] 
    # the number of residual data: (24,)
    k_res_count = ds['k_res_count'] 
    # the costs associated with the residual data: (24,)
    k_res_cost = ds['k_res_cost'] 
    # the reduced costs associated with the residual data: (24,)
    k_res_cost_reduced = ds['k_res_cost_reduced'] 
    # The mean harmonisation residual: (24,)
    k_res_mean = ds['k_res_mean'] 
    # The standard deviation of the mean harmonisation residual: (24,)
    k_res_mean_stdev = ds['k_res_mean_stdev'] 
    # The standard deviation of the harmonisation residual: (24,)
    k_res_stdev = ds['k_res_stdev'] 
    # The sensors associated with the harmonisation residual: (24,)
    k_res_sensors = ds['k_res_sensors'] 

    # The multivariate normal, multinormal or Gaussian distribution is a 
    # generalization of the 1D-normal distribution to higher dimensions. 
    # Such a distribution is specified by its mean and covariance matrix. 
    # These parameters are analogous to the mean (average or “center”) and 
    # variance (standard deviation, or “width,” squared) of the 1D-normal distribution.

    Xmean = parameter 
    Xcov = parameter_covariance_matrix
    size = 100
    draws = np.random.multivariate_normal(Xmean, Xcov, size)

    # np.random.multivariate_normal(Xmean, Xcov[, size, check_valid, tol])
    # Xmean : 1-D array_like, of length N : mean of the N-dimensional distribution
    # Xcov : 2-D array_like, of shape (N, N) : covariance matrix of the distribution (symmetric and positive-semidefinite for proper sampling)
    # size : int or tuple of ints : optional
    # check_valid : { ‘warn’, ‘raise’, ‘ignore’ } : optional (behavior when the covariance matrix is not positive semidefinite)
    # tol : float : optional (tolerance when checking the singular values in covariance matrix)
    # draws : ndarray : drawn samples, of shape size, if that was provided (if not, the shape is (N,) i.e. each entry out[i,j,...,:] is an N-dimensional value drawn from the distribution)
    # Given a shape of, for example, (m,n,k), m*n*k samples are generated, and packed in an m x n x k arrangement. 
    # Because each sample is N-dimensional, the output shape is (m,n,k,N). If no shape is specified, a single (N-D) sample is returned.

    return draws

def plot_ensemble(ds, draws):
    '''
    Plot ensemble
    '''
    # Harmonisation parameters: (27,)
    parameter = ds['parameter'] 
    # Harmonisation parameter covariance matrix: (27, 27)
    parameter_covariance_matrix = ds['parameter_covariance_matrix'] 

    X = parameter
    Y = draws
    N = len(draws)
    M = len(X)

    fig = plt.figure()
    plt.plot(Y)
    for i in range(0,N):
        for j in range(0,M):
            plt.scatter(i, X[j], marker='o', color='k')
    plt.xlabel('Ensemble member')
    plt.ylabel('Coefficient value')
    plt.title('FCDR harmonisation coefficients')
    plt.savefig('ensemble.png')    


    #
    # Histograms of ensemble coefficient variability
    #

    fig, ax = plt.subplots(9,3,sharex=False)
    for i in range(0,9):
        for j in range(0,3):
            k = (3*i)+j
            hist, bins = np.histogram(Y[:,k], bins=10, density=True) 
            a_mean = Y[:,k].mean()
            ax[i,j].plot(bins[0:-1], hist)
            ax[i,j].plot((a_mean,a_mean), (0,hist.max()), 'r-')   
            ax[i,j].tick_params(labelsize=6)
    plt.savefig('histograms.png')

if __name__ == "__main__":

#    parser = OptionParser("usage: %prog file_in")
#    (options, args) = parser.parse_args()
#    file_in = args[0]

    file_in = "harm_FO_3.0_27b18eb_ODR_101_11_R____AVHRR_REAL_4_RSA______19911004_20151231.nc"
    ds = load_data(file_in)
    draws = calc_ensemble(ds)
    plot_ensemble(ds, draws)

    eigenval, eigenvec = calc_eigen(ds)
    plot_eigenval(eigenval)
    plot_eigenvec(eigenvec)
    plot_covariance(ds)
    plot_sample_binormal()




