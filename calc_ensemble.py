#!/usr/bin/env python

# ipdb> import os; os._exit(1)

# call as: python calc_ensemble.py

# =======================================
# Version 0.7
# 2 April, 2019
# michael.taylor AT reading DOT ac DOT uk
# =======================================

import os  
import os.path  
from os import fsync, remove
import glob  
import optparse 
from  optparse import OptionParser  
import sys   
import numpy as np
import numpy.ma as ma  
from numpy import array_equal, savetxt, loadtxt, frombuffer, save as np_save, load as np_load, savez_compressed, array
import xarray
import pandas as pd
from pandas import Series, DataFrame, Panel
import scipy
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
import seaborn as sns; sns.set(style="darkgrid")
import matplotlib.pyplot as plt; plt.close("all")
import matplotlib.dates as mdates
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D

# =======================================    
# AUXILIARY METHODS
# =======================================    

def fmt(x, pos):
    '''
    Allow for expoential notation in colorbar labels
    '''

    a, b = '{0:.3e}'.format(x).split('e')
    b = int(b)

    return r'${} \times 10^{{{}}}$'.format(a, b)

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

def plot_parameters(ds, npar, sensor):
    '''
    Plot harmonisation parameters of best-case
    '''

    parameter = ds['parameter'] 

    X = np.arange(0,len(sensor))
    Y = np.array(parameter)

    if npar == 3:

        idx0 = np.arange(0, len(Y), 3)        
        idx1 = np.arange(1, len(Y), 3)        
        idx2 = np.arange(2, len(Y), 3) 
        df = pd.DataFrame()
        Y0 = []
        Y1 = []
        Y2 = []
        for i in range(0,len(X)): 
            k0 = idx0[i]
            k1 = idx1[i]
            k2 = idx2[i]
            Y0.append(Y[k0])
            Y1.append(Y[k1])
            Y2.append(Y[k2])
        Y = np.array([Y0,Y1,Y2])    
        df = pd.DataFrame({'a(0)': Y0, 'a(1)': Y1, 'a(2)': Y2}, index=list(sensor))                  
        ax = df.plot(kind="bar", subplots=True, layout=(3, 1), sharey=False, sharex=True, rot=90, fontsize=12, legend=False)

    elif npar == 4:

        idx0 = np.arange(0, len(Y), 4)        
        idx1 = np.arange(1, len(Y), 4)        
        idx2 = np.arange(2, len(Y), 4) 
        idx3 = np.arange(3, len(Y), 4) 
        df = pd.DataFrame()
        Y0 = []
        Y1 = []
        Y2 = []
        Y3 = []
        for i in range(0,len(X)): 
            k0 = idx0[i]
            k1 = idx1[i]
            k2 = idx2[i]
            k3 = idx3[i]
            Y0.append(Y[k0])
            Y1.append(Y[k1])
            Y2.append(Y[k2])
            Y3.append(Y[k3])
        Y = np.array([Y0,Y1,Y2,Y3])    
        df = pd.DataFrame({'a(0)': Y0, 'a(1)': Y1, 'a(2)': Y2, 'a(3)': Y3}, index=list(sensor))                  
        ax = df.plot(kind="bar", subplots=True, layout=(4, 1), sharey=False, sharex=True, rot=90, fontsize=12, legend=False)
    
    plt.tight_layout()
    file_str = "bestcase_parameters.png"
    plt.savefig(file_str)    

def plot_covariance(ds):
    '''
    Plot harmonisation parameter covariance matrix as a heatmap
    '''

    parameter_covariance_matrix = ds['parameter_covariance_matrix'] 

    X = np.array(parameter_covariance_matrix)
    Xmin = X.min()    
    Xmax = X.max()    
#    Y = X / Xmax
    Y = X

    fig = plt.figure()
#    log_norm = colors.LogNorm(vmin=Xmax, vmax=Xmax)
#    cbar_ticks = [(10 ** i) for i in range(-6,1)]  
#    sns.heatmap(Y, center=0.5, linewidths=.5, cmap="viridis", cbar=True, norm=log_norm, cbar_kws={"ticks":cbar_ticks})
#    sns.heatmap(Y, center=0, linewidths=.5, cmap="viridis", cbar=True, vmin=-1.0e-9, vmax=1.0e-6, cbar_kws={"extend":'both', "format":'%.3e'})

    sns.heatmap(Y, center=0, linewidths=.5, cmap="viridis", cbar=True, vmin=-1.0e-9, vmax=1.0e-6, cbar_kws={"extend":'both', "format":ticker.FuncFormatter(fmt)})

    #
    # Mask out upper triangle
    #

    # mask = np.zeros_like(X)
    # mask[np.triu_indices_from(mask)] = True
    # with sns.axes_style("white"):
    #     sns.heatmap(Y, mask=mask, square=True, center=0, linewidths=.5, cmap="viridis", cbar=True)

    title_str = "Covariance matrix: max=" + "{0:.3e}".format(Xmax)
    plt.title(title_str)
    plt.savefig('bestcase_covariance_matrix.png')    

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
    plt.savefig('bestcase_eigenvalues.png')    

def plot_eigenvec(eigenvec):
    '''
    Plot eigenvector matrix as a heatmap
    '''
    X = eigenvec

    fig = plt.figure()
    sns.heatmap(X, center=0, linewidths=.5, cmap="viridis", cbar=True)
#    sns.heatmap(X, vmin=0, vmax=1, linewidths=.5, cmap="viridis", cbar=True)
#    sns.heatmap(X, center=0, linewidths=.5, annot=True, fmt="f", cmap="viridis", cbar=True)

    plt.title('Eigenvector matrix')
    plt.savefig('bestcase_eigenvectors.png')    

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
    size = 1000000
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

    # 
    # Fast save of draws array
    #

    np_save('draws.npy', draws, allow_pickle=False)

    return draws

def plot_ensemble_histograms(ds, draws, npar, sensor):
    '''
    Plot histograms of ensemble coefficient z-scores
    '''

    parameter = ds['parameter'] 
    parameter_covariance_matrix = ds['parameter_covariance_matrix'] 

    X = parameter
    Y = draws
    N = len(draws)
    M = len(X)

    #
    # Histograms of ensemble variability
    #

    for i in range(0,len(sensor)):

        fig, ax = plt.subplots(npar,1,sharex=True)
        for j in range(0,npar):
            k = (npar*i)+j
            Y_mean = Y[:,k].mean()
            Y_std = Y[:,k].std()
            Z = (Y[:,k] - Y_mean) / Y_std
            hist, bins = np.histogram(Z, bins=100, density=False) 
            hist = hist/hist.sum()
            ax[j].fill_between(bins[0:-1], hist, step="post", alpha=0.4)
            ax[j].plot(bins[0:-1], hist, drawstyle='steps-post')
            ax[j].plot((0,0), (0,hist.max()), 'r-')   
            ax[j].tick_params(labelsize=10)
            ax[j].set_xlim([-6,6])
            ax[j].set_ylabel('Prob. density', fontsize=10)
            title_str = sensor[i] + ": a(" + str(j+1) + ")=" + "{0:.3e}".format(Y_mean)
            ax[j].set_title(title_str, fontsize=10)
            
        plt.xlabel(r'z-score', fontsize=10)
        file_str = "ensemble_histograms_" + sensor[i] + ".png"
        plt.savefig(file_str)    

def plot_ensemble_coefficients(ds, draws, npar, sensor):
    '''
    Plot ensemble coefficient z-scores
    '''

    parameter = ds['parameter'] 
    parameter_covariance_matrix = ds['parameter_covariance_matrix'] 

    X = parameter
    Y = draws
    N = len(draws)
    M = len(X)

    #
    # Ensemble coefficients plotted as z-scores relative to best value
    #

    for i in range(0,len(sensor)):

        fig, ax = plt.subplots(npar,1,sharex=True)
        for j in range(0,npar):
            k = (npar*i)+j
            Y_mean = Y[:,k].mean()
            Y_std = Y[:,k].std()
            Z =  (Y[:,k] - Y_mean) / Y_std
            ax[j].plot(np.arange(0,N),Z)            
            ax[j].plot((0,N), (0,0), 'r-')   
            ax[j].tick_params(labelsize=10)
            ax[j].set_ylabel(r'z-score', fontsize=10)
            ax[j].set_ylim([-6,6])
            title_str = sensor[i] + ": a(" + str(j+1) + ")=" + "{0:.3e}".format(Y_mean)
            ax[j].set_title(title_str, fontsize=10)

        plt.xlabel('Ensemble member', fontsize=10)
        file_str = "ensemble_coefficients_" + sensor[i] + ".png"
        plt.savefig(file_str)    

def calc_equiprobable(ds, draws, npar, sensor):
    '''
    Extract decile members
    '''

    parameter = ds['parameter'] 
    parameter_covariance_matrix = ds['parameter_covariance_matrix'] 

    X = parameter
    Y = draws
    N = len(draws)
    M = len(X)

    Y_mean = Y.mean(axis=0)
    Y_sd = Y.std(axis=0)
    Z = (Y - Y_mean) / Y_sd

    F = np.array(range(N))/float(N)    
#    Z_cdf = np.empty(shape=(N,M))
    n = 10
    ensemble = np.empty(shape=(n,M))

    for i in range(M):

        # 
        # CDF of Z-scores of draw distribution
        #

        Z_cdf = np.sort(Z[:,i])
  
        #
        # Construct ensemble from decile values of Z_cdf
        # 

        for j in range(n):

            idx = int(j * (N/n))
            ensemble[j,i] = Z_cdf[idx]

    nbins = 10
    Z_est = np.empty(shape=(nbins,M))

    fig, ax = plt.subplots()
    for i in range(M):

        #
        # Empirical CDF
        #

        hist, edges = np.histogram( Z[:,i], bins = nbins, density = True )
        binwidth = edges[1] - edges[0]
        Z_est[:,i] = np.cumsum(hist) * binwidth
        F_est = edges[1:]
        label_str = 'Ensemble member' + str(i)
        plt.plot(F_est, Z_est[:,i], label=label_str)
        plt.xlabel('z-score')
        plt.ylabel('Cumulative distribution function (CDF)')
        plt.legend()

    plt.savefig('Z_est.png')

    return ensemble

def plot_ensemble(ds, ensemble, npar, sensor):
    '''
    Plot ensemble
    '''

    parameter = ds['parameter'] 

    X = np.arange(0,len(sensor))
    Y = np.array(parameter)

    if npar == 3:

        idx0 = np.arange(0, len(Y), 3)        
        idx1 = np.arange(1, len(Y), 3)        
        idx2 = np.arange(2, len(Y), 3) 
        df = pd.DataFrame()
        Y0 = []
        Y1 = []
        Y2 = []
        for i in range(0,len(X)): 
            k0 = idx0[i]
            k1 = idx1[i]
            k2 = idx2[i]
            Y0.append(Y[k0])
            Y1.append(Y[k1])
            Y2.append(Y[k2])
        Y = np.array([Y0,Y1,Y2])    
        df = pd.DataFrame({'a(0)': Y0, 'a(1)': Y1, 'a(2)': Y2}, index=list(sensor))                  
    elif npar == 4:

        idx0 = np.arange(0, len(Y), 4)        
        idx1 = np.arange(1, len(Y), 4)        
        idx2 = np.arange(2, len(Y), 4) 
        idx3 = np.arange(3, len(Y), 4) 
        df = pd.DataFrame()
        Y0 = []
        Y1 = []
        Y2 = []
        Y3 = []
        for i in range(0,len(X)): 
            k0 = idx0[i]
            k1 = idx1[i]
            k2 = idx2[i]
            k3 = idx3[i]
            Y0.append(Y[k0])
            Y1.append(Y[k1])
            Y2.append(Y[k2])
            Y3.append(Y[k3])
        Y = np.array([Y0,Y1,Y2,Y3])    
        df = pd.DataFrame({'a(0)': Y0, 'a(1)': Y1, 'a(2)': Y2, 'a(3)': Y3}, index=list(sensor))                  

    
    for j in range(0,npar):
        k = (npar*i)+j


    for i in range(0,npar):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        colors = ['r', 'g', 'b', 'y']
        yticks = [3, 2, 1, 0]
        for c, k in zip(colors, yticks):
            # Generate the random data for the y=k 'layer'.
            xs = np.arange(20)
            ys = np.random.rand(20)

            # You can provide either a single color or an array with the same length as
            # xs and ys. To demonstrate this, we color the first bar of each set cyan.
            cs = [c] * len(xs)
            cs[0] = 'c'

            # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
            ax.bar(xs, ys, zs=k, zdir='y', color=cs, alpha=0.8)

        # On the y axis let's only label the discrete values that we have data for.
        ax.set_yticks(yticks)

        ax.set_xlabel('Sensor', fontsize=10)
        ax.set_ylabel('Ensemble member', fontsize=10)
        z_str = 'Coefficient: a(' + str(i) + ')'
        ax.set_zlabel(z_str, fontsize=10)
        file_str = "ensemble_coefficient_" + str(i) + ".png"
        plt.savefig(file_str)    

# =======================================    
# MAIN BLOCK
# =======================================    
    
if __name__ == "__main__":

#    parser = OptionParser("usage: %prog file_in npar")
#    (options, args) = parser.parse_args()
#    file_in = args[0]
#    npar = args[1]

#    file_in = "FIDUCEO_Harmonisation_Data_37.nc"
#    file_in = "FIDUCEO_Harmonisation_Data_11.nc"
    file_in = "FIDUCEO_Harmonisation_Data_12.nc"
#    npar = 3
    npar = 4

    sensor = ['METOPA','NOAA19','NOAA18','NOAA17','NOAA16','NOAA15','NOAA14','NOAA12','NOAA11']

    ds = load_data(file_in)
    draws = calc_ensemble(ds)

    # 
    # Fast load of draws array
    #

    # draws = np_load('draws.npy')

    ensemble = calc_equiprobable(ds, draws, npar, sensor)
    plot_ensemble_coefficients(ds, draws, npar, sensor)
    plot_ensemble_histograms(ds, draws, npar, sensor)
#    plot_ensemble(ds, ensemble, npar, sensor)

    plot_parameters(ds, npar, sensor)
    plot_covariance(ds)
    eigenval, eigenvec = calc_eigen(ds)
    plot_eigenval(eigenval)
    plot_eigenvec(eigenvec)


