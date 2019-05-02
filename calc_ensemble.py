#!/usr/bin/env python

# ipdb> import os; os._exit(1)

# call as: python calc_ensemble.py

# =======================================
# Version 0.5
# 2 May, 2019
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
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

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

    ds = xarray.open_dataset(file_in)

    return ds

def calc_eigen(X):
    '''
    Calculate eigenvalues and eigenvectors from the covariance (or correlation) matrix X
    '''

    eigenval, eigenvec = np.linalg.eig(X)

    return eigenval, eigenvec

def calc_draws(ds, npop):
    '''
    Sample from the N-normal distribution using the harmonisation parameters as the mean values (best case) and the covariance matrix as the N-variance
    '''

    # Harmonisation parameters: (nsensor x npar,)
    parameter = ds['parameter'] 
    # Harmonisation parameter uncertainties: (nsensor x npar,)
    parameter_uncertainty = ds['parameter_uncertainty'] 
    # Harmonisation parameter covariance matrix: (nsensor x npar, nsensor x npar)
    parameter_covariance_matrix = ds['parameter_covariance_matrix'] 
    # Harmonisation parameter correlation matrix: (nsensor x npar, nsensor x npar)
    parameter_correlation_matrix = ds['parameter_correlation_matrix'] 
    # Harmonisation parameter Hessian matrix (internal): (nsensor x npar, nsensor x npar)
    parameter_hessian_matrix = ds['parameter_hessian_matrix'] 
    # Harmonisation parameter add offsets (internal): (nsensor x npar,)
    parameter_add_offset = ds['parameter_add_offset'] 
    # Harmonisation parameter scale factors (internal): (nsensor x npar,)
    parameter_scale_factor = ds['parameter_scale_factor'] 
    # Sensors associated with harmonisation parameters: (nsensor x npar,)
    parameter_sensors = ds['parameter_sensors'] 
    # the number of residual data: ((nsensor-1) x npar,)
    k_res_count = ds['k_res_count'] 
    # the costs associated with the residual data: ((nsensor-1) x npar,)
    k_res_cost = ds['k_res_cost'] 
    # the reduced costs associated with the residual data: ((nsensor-1) x npar,)
    k_res_cost_reduced = ds['k_res_cost_reduced'] 
    # The mean harmonisation residual: ((nsensor-1) x npar,)
    k_res_mean = ds['k_res_mean'] 
    # The standard deviation of the mean harmonisation residual: ((nsensor-1) x npar,)
    k_res_mean_stdev = ds['k_res_mean_stdev'] 
    # The standard deviation of the harmonisation residual: ((nsensor-1) x npar,)
    k_res_stdev = ds['k_res_stdev'] 
    # The sensors associated with the harmonisation residual: ((nsensor-1) x npar,)
    k_res_sensors = ds['k_res_sensors'] 

    # The multivariate normal, multinormal or Gaussian distribution is a 
    # generalization of the 1D-normal distribution to higher dimensions. 
    # Such a distribution is specified by its mean and covariance matrix.
    # These parameters are analogous to the mean (average or “center”) and
    # variance (standard deviation, or “width,” squared) of the 1D-normal distribution.

    X_ave = parameter 
    X_cov = parameter_covariance_matrix
    size = npop
    draws = np.random.multivariate_normal(X_ave, X_cov, size)

    # np.random.multivariate_normal(X_ave, X_cov[, size, check_valid, tol])
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

def calc_ensemble(ds, draws, npar, sensor, nens, npop):
    '''
    Extract (decile) ensemble members
    '''

    parameter = ds['parameter'] 
    parameter_uncertainty = ds['parameter_uncertainty'] 

    draws_ave = draws.mean(axis=0)
    draws_std = draws.std(axis=0)

    Z = (draws - draws_ave) / draws_std
    Z_min = Z.min(axis=0)
    Z_max = Z.max(axis=0)

    print("best_mean - mean(draws):", (parameter - draws_ave) )
    print("best_uncertainty - std(draws):", (parameter_uncertainty - draws_std) )
    print("Z(draws)_min:", Z_min)
    print("Z(draws)_max:", Z_max)

    #
    # Extract deciles for each parameter from CDF of draws
    #

    decile = np.empty(shape=(nens,len(parameter)))
    decile_idx = np.empty(shape=(nens,len(parameter)))

    for i in range(0,len(parameter)):

        # 
        # CDF (+ sort indices) of draw distribution (for each parameter)
        #

        Z_cdf = np.sort(Z[:,i])
        i_cdf = np.argsort(Z[:,i])

#        --------------------------------
#        NUMPY: sort --> unsort example
#        --------------------------------
#        arr = np.array([5, 3, 7, 2, 46, 3]) 
#        arr_sorted = np.sort(arr)
#        arr_idx = np.argsort(arr)
#        arr_remade = np.empty(arr.shape)
#        arr_remade[arr_idx] = arr_sorted
#        --------------------------------
  
        #
        # Decile values of Z_cdf (for each parameter)
        # 

        idx_step = int(npop / (nens - 1))
#        deciles = np.linspace(0, nens, nens+1, endpoint=True).astype('int') * 10

        for j in range(nens):

            if j == 0:
                idx = idx_step * j
            else:
                idx = idx_step * j - 1

            decile[j,i] = Z_cdf[idx]
            decile_idx[j,i] = i_cdf[idx]            

    decile_idx = decile_idx.astype(int)
    decile_min = decile.min(axis=0)
    decile_max = decile.max(axis=0)

    print("parameter deciles: ", decile)
    print("parameter deciles(min): ", decile_min)
    print("parameter deciles(max): ", decile_max)

    #
    # Calcaulte norm of draw deltas with respect to deciles to select ensemble members
    #

    Z_norm = np.empty(shape=(nens, npop))
    for i in range(0,npop): 

        for j in range(0,nens):
            Z_norm[j,i] = np.linalg.norm( Z[i,:] - decile[j,:] )
    
    ensemble = np.empty(shape=(nens,len(parameter)))
    ensemble_idx = np.empty(shape=(nens))

    for j in range(0,nens):

#        y = np.percentile(Z_norm, deciles[j+1], interpolation='nearest') 
#        i = abs(Z_norm - y).argmin()        
        
        i = np.argmin(Z_norm[j,:])
        ensemble[j,:] = draws[i,:]
        ensemble_idx[j] = i
        
    ensemble_ave = ensemble.mean(axis=0)
    ensemble_std = ensemble.std(axis=0)
    ensemble_min = ensemble.min(axis=0)
    ensemble_max = ensemble.max(axis=0)

    print("best_mean - mean(ensemble):", (parameter - ensemble_ave) )
    print("best_uncertainty - std(ensemble):", (parameter_uncertainty - ensemble_std) )
    print("Z(ensemble)_min:", ensemble_min)
    print("Z(ensemble)_max:", ensemble_max)

    return ensemble, ensemble_idx

def plot_bestcase_parameters(ds, npar, sensor):
    '''
    Plot harmonisation parameters and uncertainties for best-case
    '''

    parameter = ds['parameter'] 
    parameter_uncertainty = ds['parameter_uncertainty'] 

    Y = np.array(parameter)
    U = np.array(parameter_uncertainty)

    if npar == 3:

#        Z = a.reshape((len(Y), npar))

        idx0 = np.arange(0, len(Y), 3)        
        idx1 = np.arange(1, len(Y), 3)        
        idx2 = np.arange(2, len(Y), 3) 
        Y0 = []
        Y1 = []
        Y2 = []
        U0 = []
        U1 = []
        U2 = []
        for i in range(0,len(sensor)): 

            k0 = idx0[i]
            k1 = idx1[i]
            k2 = idx2[i]
            Y0.append(Y[k0])
            Y1.append(Y[k1])
            Y2.append(Y[k2])
            U0.append(U[k0])
            U1.append(U[k1])
            U2.append(U[k2])
        Y = np.array([Y0,Y1,Y2])    
        U = np.array([U0,U1,U2])    
        dY = pd.DataFrame({'a(0)': Y0, 'a(1)': Y1, 'a(2)': Y2}, index=list(sensor))                  
        dU = pd.DataFrame({'a(0)': U0, 'a(1)': U1, 'a(2)': U2}, index=list(sensor)) 

        ax = dY.plot(kind="bar", yerr=dU, colormap='viridis', subplots=True, layout=(3, 1), sharey=False, sharex=True, rot=90, fontsize=12, legend=False)

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
        U0 = []
        U1 = []
        U2 = []
        U3 = []
        for i in range(0,len(sensor)): 

            k0 = idx0[i]
            k1 = idx1[i]
            k2 = idx2[i]
            k3 = idx3[i]
            Y0.append(Y[k0])
            Y1.append(Y[k1])
            Y2.append(Y[k2])
            Y3.append(Y[k3])
            U0.append(U[k0])
            U1.append(U[k1])
            U2.append(U[k2])
            U3.append(U[k3])
        Y = np.array([Y0,Y1,Y2,Y3])    
        U = np.array([U0,U1,U2,U3])    
        dY = pd.DataFrame({'a(0)': Y0, 'a(1)': Y1, 'a(2)': Y2, 'a(3)': Y3}, index=list(sensor))                  
        dU = pd.DataFrame({'a(0)': U0, 'a(1)': U1, 'a(2)': U2, 'a(3)': U3}, index=list(sensor))                  

        ax = dY.plot(kind="bar", yerr=dU, colormap='viridis', subplots=True, layout=(4, 1), sharey=False, sharex=True, rot=90, fontsize=12, legend=False)
    
    plt.tight_layout()
    file_str = "bestcase_parameters.png"
    plt.savefig(file_str)    
    plt.close()

def plot_bestcase_covariance(ds):
    '''
    Plot harmonisation parameter covariance matrix for best-case as a heatmap
    '''

    parameter_covariance_matrix = ds['parameter_covariance_matrix'] 

    Y = np.array(parameter_covariance_matrix)

    fig = plt.figure()
    sns.heatmap(Y, center=0, linewidths=.5, cmap="viridis", cbar=True, vmin=-1.0e-9, vmax=1.0e-6, cbar_kws={"extend":'both', "format":ticker.FuncFormatter(fmt)})
    title_str = "Covariance matrix (relative): max=" + "{0:.3e}".format(Y.max())
    plt.title(title_str)
    plt.savefig('bestcase_covariance_matrix.png')    
    plt.close()

def plot_population_histograms(ds, draws, npar, sensor, nens):
    '''
    Plot histograms of population coefficient z-scores
    '''

    parameter = ds['parameter'] 
    parameter_covariance_matrix = ds['parameter_covariance_matrix'] 

    draws_ave = draws.mean(axis=0)
    draws_std = draws.std(axis=0)
    Z = (draws - draws_ave) / draws_std
#    nbins = nens
    nbins = 100

    #
    # Histograms of ensemble variability
    #

    for i in range(0,len(sensor)):

        fig, ax = plt.subplots(npar,1,sharex=True)
        for j in range(0,npar):

            k = (npar*i)+j
            hist, bins = np.histogram(Z[:,k], bins=nbins, density=False) 
            hist = hist/hist.sum()
            ax[j].fill_between(bins[0:-1], hist, step="post", alpha=0.4)
            ax[j].plot(bins[0:-1], hist, drawstyle='steps-post')
            ax[j].plot((0,0), (0,hist.max()), 'r-')   
            ax[j].tick_params(labelsize=10)
            ax[j].set_xlim([-6,6])
            ax[j].set_ylabel('Prob. density', fontsize=10)
            title_str = sensor[i] + ": a(" + str(j) + ")=" + "{0:.3e}".format(draws_ave[k])
            ax[j].set_title(title_str, fontsize=10)
            
        plt.xlabel(r'z-score', fontsize=10)
        file_str = "population_histograms_" + sensor[i] + ".png"
        plt.savefig(file_str)    

    plt.close('all')

def plot_population_coefficients(ds, draws, npar, sensor, npop):
    '''
    Plot population coefficient z-scores
    '''

    parameter = ds['parameter'] 
    parameter_covariance_matrix = ds['parameter_covariance_matrix'] 

    draws_ave = draws.mean(axis=0)
    draws_std = draws.std(axis=0)
    Z = (draws - draws_ave) / draws_std

    #
    # Ensemble coefficients plotted as z-scores relative to best value
    #

    for i in range(0,len(sensor)):

        fig, ax = plt.subplots(npar,1,sharex=True)
        for j in range(0,npar):

            k = (npar*i)+j
            ax[j].plot(np.arange(0,npop), Z[:,k])            
            ax[j].plot((0,npop), (0,0), 'r-')   
            ax[j].tick_params(labelsize=10)
            ax[j].set_ylabel(r'z-score', fontsize=10)
            ax[j].set_ylim([-6,6])
            title_str = sensor[i] + ": a(" + str(j) + ")=" + "{0:.3e}".format(draws_ave[k])
            ax[j].set_title(title_str, fontsize=10)

        plt.xlabel('Population member', fontsize=10)
        file_str = "population_coefficients_" + sensor[i] + ".png"
        plt.savefig(file_str)    

    plt.close('all')

def plot_population_cdf(ds, draws, npar, sensor, nens, npop):
    '''
    Extract decile values of population
    '''

    parameter = ds['parameter'] 

    draws_ave = draws.mean(axis=0)
    draws_std = draws.std(axis=0)
    Z = (draws - draws_ave) / draws_std

    F = np.array(range(0,npop))/float(npop)    
    Z_est = np.empty(shape=(nens,len(parameter)))

    for i in range(0,npar):

        fig, ax = plt.subplots()
        for j in range(0,len(sensor)):

            k = (npar*i)+j

            #
            # Empirical CDF
            #

            hist, edges = np.histogram( Z[:,k], bins = nens, density = True )
            binwidth = edges[1] - edges[0]
            Z_cdf = np.sort(Z[:,k])
            Z_est[:,k] = np.cumsum(hist) * binwidth
            F_est = edges[1:]
            label_str = sensor[j]
            plt.plot(F_est, Z_est[:,k], marker='.', linewidth=0.25, label=label_str)
            plt.xlim([-6,6])
            plt.ylim([0,1])
            plt.xlabel('z-score')
            plt.ylabel('Cumulative distribution function (CDF)')
            title_str = 'Harmonisation coefficient: a(' + str(i) + ')' 
            plt.title(title_str)
            plt.legend(fontsize=10, ncol=1)
            file_str = "population_cdf_coefficient_" + str(i) + ".png"
            plt.savefig(file_str)    

    plt.close('all')

def plot_ensemble_deltas(ds, ensemble, npar, sensor, nens):
    '''
    Plot ensemble member coefficients normalised to parameter uncertainty
    '''

    parameter = ds['parameter'] 
    parameter_uncertainty = ds['parameter_uncertainty'] 

    Y = np.array(parameter)
    U = np.array(parameter_uncertainty)
    Z = np.array(ensemble)

    if npar == 3:

        idx0 = np.arange(0, len(Y), 3)        
        idx1 = np.arange(1, len(Y), 3)        
        idx2 = np.arange(2, len(Y), 3) 
        Y0 = []
        Y1 = []
        Y2 = []
        U0 = []
        U1 = []
        U2 = []
        Z0 = []
        Z1 = []
        Z2 = []
        for i in range(0,len(sensor)): 

            k0 = idx0[i]
            k1 = idx1[i]
            k2 = idx2[i]
            Y0.append(Y[k0])
            Y1.append(Y[k1])
            Y2.append(Y[k2])
            U0.append(U[k0])
            U1.append(U[k1])
            U2.append(U[k2])
            Z0.append(Z[:,k0])
            Z1.append(Z[:,k1])
            Z2.append(Z[:,k2])
        Y = np.array([Y0,Y1,Y2])    
        U = np.array([U0,U1,U2])    
        Z = np.array([Z0,Z1,Z2])    
        dY = pd.DataFrame({'a(0)': Y0, 'a(1)': Y1, 'a(2)': Y2}, index=list(sensor))        
        dU = pd.DataFrame({'a(0)': U0, 'a(1)': U1, 'a(2)': U2}, index=list(sensor))             
        dZ = pd.DataFrame({'a(0)': Z0, 'a(1)': Z1, 'a(2)': Z2}, index=list(sensor))                  

    elif npar == 4:

        idx0 = np.arange(0, len(Y), 4)        
        idx1 = np.arange(1, len(Y), 4)        
        idx2 = np.arange(2, len(Y), 4) 
        idx3 = np.arange(3, len(Y), 4) 
        Y0 = []
        Y1 = []
        Y2 = []
        Y3 = []
        U0 = []
        U1 = []
        U2 = []
        U3 = []
        Z0 = []
        Z1 = []
        Z2 = []
        Z3 = []
        for i in range(0,len(sensor)): 

            k0 = idx0[i]
            k1 = idx1[i]
            k2 = idx2[i]
            k3 = idx3[i]
            Y0.append(Y[k0])
            Y1.append(Y[k1])
            Y2.append(Y[k2])
            Y3.append(Y[k3])
            U0.append(U[k0])
            U1.append(U[k1])
            U2.append(U[k2])
            U3.append(U[k3])
            Z0.append(Z[:,k0])
            Z1.append(Z[:,k1])
            Z2.append(Z[:,k2])
            Z3.append(Z[:,k3])
        Y = np.array([Y0,Y1,Y2,Y3])    
        U = np.array([U0,U1,U2,U3])    
        Z = np.array([Z0,Z1,Z2,Z3])    
        dY = pd.DataFrame({'a(0)': Y0, 'a(1)': Y1, 'a(2)': Y2, 'a(3)': Y3}, index=list(sensor))
        dU = pd.DataFrame({'a(0)': U0, 'a(1)': U1, 'a(2)': U2, 'a(3)': U3}, index=list(sensor)) 
        dZ = pd.DataFrame({'a(0)': Z0, 'a(1)': Z1, 'a(2)': Z2, 'a(3)': Z3}, index=list(sensor))              

    #
    # Lineplot per sensor of ensemble for each parameter
    #

    xs = np.arange(nens)
    for i in range(0,npar):

        fig, ax = plt.subplots()
        for j in range(0,len(sensor)):
                  
            if i == 0:
                ys = (dZ['a(0)'][j] - dY['a(0)'][j]) / dU['a(0)'][j]
            elif i == 1:
                ys = (dZ['a(1)'][j] - dY['a(1)'][j]) / dU['a(1)'][j] 
            elif i == 2:
                ys = (dZ['a(2)'][j] - dY['a(2)'][j]) / dU['a(2)'][j] 
            elif i == 3:
                ys = (dZ['a(3)'][j] - dY['a(3)'][j]) / dU['a(3)'][j] 
            plt.plot(xs, np.sort(ys), marker='.', linewidth=0.5, label=sensor[j])

        ax.set_xlabel('Ensemble member', fontsize=12)
        ax.set_ylabel('Delta / Uncertainty', fontsize=12)
        ax.set_ylim([-6,6])
        title_str = 'Harmonisation coefficient: a(' + str(i) + ')'
        ax.set_title(title_str, fontsize=12)
        ax.legend(fontsize=8)
        file_str = "ensemble_delta_uncertainty_coefficient_" + str(i) + ".png"
        plt.savefig(file_str)    

    plt.close('all')

    #
    # Boxplot per sensor of ensemble for each parameter
    #

    xs = np.arange(nens)
    for i in range(0,npar):

        fig = plt.figure()
        ax = plt.subplot(111)
        ys_all = []
        for j in range(0,len(sensor)):
                  
            if i == 0:
                ys = (dZ['a(0)'][j] - dY['a(0)'][j]) / dU['a(0)'][j]
            elif i == 1:
                ys = (dZ['a(1)'][j] - dY['a(1)'][j]) / dU['a(1)'][j] 
            elif i == 2:
                ys = (dZ['a(2)'][j] - dY['a(2)'][j]) / dU['a(2)'][j] 
            elif i == 3:
                ys = (dZ['a(3)'][j] - dY['a(3)'][j]) / dU['a(3)'][j] 
            ys_all.append(ys)

        ax.boxplot(ys_all, notch=False, sym="o", labels=list(sensor))
        ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
        ax.set_ylabel('Delta / Uncertainty (median statistics)', fontsize=12)
        ax.set_ylim([-6,6])
        title_str = 'Harmonisation coefficient: a(' + str(i) + ')'
        ax.set_title(title_str, fontsize=12)
        file_str = "ensemble_boxplot_delta_uncertainty_coefficient_" + str(i) + ".png"
#        plt.gcf().subplots_adjust(bottom=0.15)
        plt.tight_layout()
        plt.savefig(file_str)    

    plt.close('all')

    #
    # Correlation between ensemble member coefficients
    #

    for i in range(0,npar):

        r = np.empty(shape=(len(sensor),len(sensor)))

        for j in range(0,len(sensor)):

            for k in range(0,len(sensor)):
                if i == 0:
                    yj = dZ['a(0)'][j]
                    yk = dZ['a(0)'][k]
                elif i == 1:
                    yj = dZ['a(1)'][j]
                    yk = dZ['a(1)'][k]
                elif i == 2:
                    yj = dZ['a(2)'][j]
                    yk = dZ['a(2)'][k]
                elif i == 3:
                    yj = dZ['a(3)'][j]
                    yk = dZ['a(3)'][k]

                r[j,k] = np.corrcoef(np.sort(yj), np.sort(yk))[0,1]
                r[k,j] = np.corrcoef(np.sort(yj), np.sort(yk))[1,0]

        fig, ax = plt.subplots()

        #
        # Mask out upper triangle
        #

        mask = np.zeros_like(r)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):

            g = sns.heatmap(r, mask=mask, square=False, annot=True, linewidths=0.5, cmap="viridis", cbar=True, cbar_kws={'label': 'Correlation Coeff.', 'orientation': 'vertical'}, xticklabels=sensor, yticklabels=sensor)
            g.set_yticklabels(g.get_yticklabels(), rotation =0)
            g.set_xticklabels(g.get_yticklabels(), rotation =90)
            title_str = 'Harmonisation coefficient: a(' + str(i) + ')'
            ax.set_title(title_str, fontsize=12)
            plt.tight_layout()
            file_str = "ensemble_correlation_coefficient_" + str(i) + ".png"
            plt.savefig(file_str)    

    plt.close('all')

def plot_eigenval(eigenval, title_str, file_str):
    '''
    Plot eigenvalues as a scree plot
    '''

    Y = eigenval / max(eigenval)

    fig = plt.figure()
    plt.fill_between( np.arange(0,len(Y)), Y, step="post", alpha=0.4)
    plt.plot( np.arange(0,len(Y)), Y, drawstyle='steps-post')
    plt.tick_params(labelsize=12)
    plt.ylabel("Relative value", fontsize=12)
#    title_str = 'Scree plot: eigenvalue max=' + "{0:.5f}".format(eigenval.max())
#    file_str = 'bestcase_eigenvalues.png'
    plt.title(title_str)
    plt.savefig(file_str)
    plt.close()

def plot_eigenvec(eigenvec, title_str, file_str):
    '''
    Plot eigenvector matrix as a heatmap
    '''

    X = eigenvec

    fig = plt.figure()
    sns.heatmap(X, center=0, linewidths=.5, cmap="viridis", cbar=True)
#    title_str = 'Eigenvector matrix'
#    file_str = 'bestcase_eigenvectors.png'
    plt.title(title_str)
    plt.savefig(file_str)
    plt.close()

def plot_ensemble_check(ds, ensemble):
    '''
    Calculate correlation matrix of ensemble.
    Calculate covariance matrix of ensemble.
    Calculate diff from harmonisation.
    Eigenvalue scree plot of covariance and correlation matrices.
    '''
    cov_par = ds.parameter_covariance_matrix
    cov_ensemble = np.cov(ensemble, rowvar=False)
    cov_diff = cov_par - cov_ensemble

    corr_par = ds.parameter_correlation_matrix
    corr_ensemble = np.corrcoef(ensemble, rowvar=False)
    corr_diff = corr_par - corr_ensemble

    cov_par_eigenval, cov_par_eigenvec = calc_eigen(cov_par)
    cov_ensemble_eigenval, cov_ensemble_eigenvec = calc_eigen(cov_ensemble)

    corr_par_eigenval, corr_par_eigenvec = calc_eigen(corr_par)
    corr_ensemble_eigenval, corr_ensemble_eigenvec = calc_eigen(corr_ensemble)

    #
    # Plot Eigenvalues
    #

    title_str = 'Scree plot (correlation matrix: harmonisation): eigenvalue max=' + "{0:.5f}".format(corr_par_eigenval.max())
    file_str = 'harmonisation_eigenvalues_correlation_matrix.png'
    plot_eigenval(corr_par_eigenval, title_str, file_str) 

    title_str = 'Scree plot (correlation matrix: ensemble): eigenvalue max=' + "{0:.5f}".format(corr_ensemble_eigenval.max())
    file_str = 'ensemble_eigenvalues_correlation_matrix.png'
    plot_eigenval(corr_ensemble_eigenval, title_str, file_str) 

    title_str = 'Scree plot (covariance matrix: harmonisation): eigenvalue max=' + "{0:.5f}".format(cov_par_eigenval.max())
    file_str = 'harmonisation_eigenvalues_covariance_matrix.png'
    plot_eigenval(cov_par_eigenval, title_str, file_str) 

    title_str = 'Scree plot (covariance matrix: ensemble): eigenvalue max=' + "{0:.5f}".format(cov_ensemble_eigenval.max())
    file_str = 'ensemble_eigenvalues_covariance_matrix.png'
    plot_eigenval(cov_ensemble_eigenval, title_str, file_str) 

    #
    # Plot Correlation Matrices
    #

    title_str = 'Correlation matrix (harmonisation)'
    file_str = 'harmonisation_correlation_matrix.png'
    plot_eigenvec(corr_par, title_str, file_str) 

    title_str = "Correlation matrix (ensemble)"
    file_str = 'ensemble_correlation_matrix.png'    
    plot_eigenvec(corr_ensemble, title_str, file_str) 

    title_str = "Correlation matrix (harmonisation - ensemble)"
    file_str = 'diff_correlation_matrix.png'
    plot_eigenvec(corr_diff, title_str, file_str) 

    #
    # Plot Covariance Matrices
    #

    title_str = 'Covariance matrix (harmonisation)'
    file_str = 'harmonisation_covariance_matrix.png'
    plot_eigenvec(cov_par, title_str, file_str) 

    title_str = "Covariance matrix (ensemble)"
    file_str = 'ensemble_covariance_matrix.png'    
    plot_eigenvec(cov_ensemble, title_str, file_str) 

    title_str = "Covariance matrix (harmonisation - ensemble)"
    file_str = 'diff_covariance_matrix.png'
    plot_eigenvec(cov_diff, title_str, file_str)     

def export_ensemble(ensemble):
    '''
    Export ensemble in format needed for FCDR delta creation algorithm in FCDR generation code:
    1) netCDF4 format produced for Harmonisation parameter deltas (fill_val = 0) at: /gws/nopw/j04/fiduceo/Data/FCDR/AVHRR/test/MC_Harmonisation.nc. 
    2) FCDR generation code uses an environment variable (FIDUCEO_MC_HARM) to find MC_Harmonisation.nc
    3) MC_harmonisation.nc contains  metadata which enables a match to the original Harmonisation file in: /gws/nopw/j04/fiduceo/Data/FCDR/AVHRR/test/
    4) Each input harmonisation file has a UUID which is then stored as HARM_UUID3 or HARM_UUID4 or HARM_UUID5 in the ensemble delta file. The UUIDs all have to match for the file to be read in - enforcing a match to ensure that the same harmonisation files and the ensemble delta file are compatible
    5) Relevant harmonisation parameter files contain parameter names (e.g. delta_param3) are in the same directory: /gws/nopw/j04/fiduceo/Data/FCDR/AVHRR/test/
    6) Ensemble generation code will read in ensemble output file provided here
    '''    

# =======================================    
# MAIN BLOCK
# =======================================    
    
if __name__ == "__main__":

#    parser = OptionParser("usage: %prog file_in npar npop nens")
#    (options, args) = parser.parse_args()
#    file_in = args[0]
#    npar = args[1]
#    npop = args[2]
#    nens = args[3]

    file_in = "FIDUCEO_Harmonisation_Data_37.nc"
#    file_in = "FIDUCEO_Harmonisation_Data_11.nc"
#    file_in = "FIDUCEO_Harmonisation_Data_12.nc"
    npar = 3
#    npar = 4
    npop = 1000000
    nens = 11
    sensor = ['METOPA','NOAA19','NOAA18','NOAA17','NOAA16','NOAA15','NOAA14','NOAA12','NOAA11']

    ds = load_data(file_in)
#    draws = calc_draws(ds, npop)

    # 
    # Fast load of draws array
    #

    draws = np_load('draws_37_1000000.npy')
#    draws = np_load('draws_11_1000000.npy')
#    draws = np_load('draws_12_1000000.npy')

    ensemble, ensemble_idx = calc_ensemble(ds, draws, npar, sensor, nens, npop)

    plot_ensemble_check(ds, ensemble)
    plot_ensemble_deltas(ds, ensemble, npar, sensor, nens)

    plot_bestcase_parameters(ds, npar, sensor)
    plot_bestcase_covariance(ds)
    plot_population_coefficients(ds, draws, npar, sensor, npop)
    plot_population_histograms(ds, draws, npar, sensor, nens)
    plot_population_cdf(ds, draws, npar, sensor, nens, npop)

#    export_ensemble(ensemble)



