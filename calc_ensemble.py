#!/usr/bin/env python

# call as: python calc_ensemble.py
# NB: include code: plot_cov2ensemble.py

# =======================================
# Version 0.27
# 24 July, 2019
# https://patternizer.github.io/  
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
from numpy import array_equal, savetxt, loadtxt, frombuffer, save as np_save, load as np_load, savez_compressed, array
from netCDF4 import Dataset
import xarray
import scipy
import seaborn as sns; sns.set(style="darkgrid")
import matplotlib.pyplot as plt; plt.close("all")

#------------------------------------------------------------------------------
import cov2u as cov2u        # estimation of uncertainty from covariance matrix
import convert_func as con   # measurement equations & L<-->BT conversion
#------------------------------------------------------------------------------

# =======================================    
# AUXILIARY METHODS
# =======================================    

def FPE(x0,x1):
    '''
    Calculate the fractional percentage error between two arrays
    '''

    FPE = 100.*(1.-np.linalg.norm(x0)/np.linalg.norm(x1))

    return FPE

def fmt(x, pos):
    '''
    Allow for expoential notation in colorbar labels
    '''

    a, b = '{0:.3e}'.format(x).split('e')
    b = int(b)

    return r'${} \times 10^{{{}}}$'.format(a, b)

def find_nearest(array, value):
    '''
    Find value and idx of value closest to target value in array
    '''

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()

    return array[idx], idx

def calc_eigen(x):
    '''
    Calculate eigenvalues and eigenvectors from the covariance (or correlation) matrix X
    '''

    x = np.matrix(x)
    eigenval, eigenvec = np.linalg.eig(x)

    return eigenval, eigenvec

def calc_svd(x):
    '''
    Calculate singular value decomposition of  the covariance (or correlation) matrix X
    '''

    x = np.matrix(x)
    u, s, v = np.linalg.svd(X, full_matrices=True)

    return u, s, v

# =======================================    

def calc_draws(ds, npop):
    '''
    Sample from the N-normal distribution using the harmonisation parameters as the mean values (best case) and the covariance matrix as the N-variance

    # The multivariate normal, multinormal or Gaussian distribution is a 
    # generalization of the 1D-normal distribution to higher dimensions. 
    # Such a distribution is specified by its mean and covariance matrix.
    # These parameters are analogous to the mean (average or “center”) and
    # variance (standard deviation, or “width,” squared) of the 1D-normal distribution.

    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.multivariate_normal.html

    # Harmonisation parameters: (nsensor x npar,)
    # Harmonisation parameter uncertainties: (nsensor x npar,)
    # Harmonisation parameter covariance matrix: (nsensor x npar, nsensor x npar)
    # Harmonisation parameter correlation matrix: (nsensor x npar, nsensor x npar)
    # Harmonisation parameter add offsets (internal): (nsensor x npar,)
    # Harmonisation parameter scale factors (internal): (nsensor x npar,)
    # Sensors associated with harmonisation parameters: (nsensor x npar,)
    '''

    X_ave = ds['parameter'] 
    X_cov = ds['parameter_covariance_matrix'] 
    X_cor = ds['parameter_correlation_matrix'] 
    X_u = ds['parameter_uncertainty'] 

    draws = np.random.multivariate_normal(X_ave, X_cov, npop)

    return draws

def calc_ensemble(ds, draws, nens, npop):
    '''
    Extract (decile) ensemble members
    '''

    parameter = ds['parameter'] 

    draws_ave = draws.mean(axis=0)
    draws_std = draws.std(axis=0)
    Z = (draws - draws_ave) / draws_std

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
  
        #
        # Decile values of Z_cdf (for each parameter): use decile mid-points
        # 

        idx = (np.linspace(0, (npop-(npop/nens))-1, nens, endpoint=True) + (npop/nens)/2).astype('int')

        for j in range(len(idx)):
            decile[j,i] = Z_cdf[idx[j]]
            decile_idx[j,i] = i_cdf[idx[j]]            
        decile_idx = decile_idx.astype(int)

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
        # y = np.percentile(Z_norm, deciles[j+1], interpolation='nearest') 
        # i = abs(Z_norm - y).argmin()        
        idx = np.argmin(Z_norm[j,:])
        ensemble[j,:] = draws[idx,:]
        ensemble_idx[j] = idx
    ensemble_idx = ensemble_idx.astype(int)

    ens = {}
    ens['ensemble'] = ensemble
    ens['ensemble_idx'] = ensemble_idx
    ens['decile'] = decile
    ens['decile_idx'] = decile_idx
    ens['Z'] = Z
    ens['Z_norm'] = Z_norm
    
    return ens
       
# =======================================    
# INCLUDE PLOT CODE:
exec(open('plot_cov2ensemble.py').read())
# =======================================    

# =======================================    
# MAIN BLOCK
# =======================================    
    
if __name__ == "__main__":

    #--------------------------------------------------------------------------
    # parser = OptionParser("usage: %prog ch npop nens har_file ens_file")
    # (options, args) = parser.parse_args()
    #--------------------------------------------------------------------------
    # ch = args[0]
    # npop = args[1]
    # nens = args[2]
    # har_file = args[3]
    # ens_file = args[4]
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    # RUN PARAMETERS:
    #--------------------------------------------------------------------------
    ch = 37
    npop = 1000000
    nens = 10
    har_file = 'FIDUCEO_Harmonisation_Data_' + str(ch) + '.nc'
    ens_file = 'MC_Harmonisation_MNS.nc'

    FLAG_load_draws = 1
    FLAG_load_ensemble = 1
    FLAG_write_netcdf = 0
    FLAG_plot = 0
#    software_tag = '3e8c463' # job dir=job_avhxx_v6_EIV_10x_11 (old runs)
#    software_tag = '4d111a1' # job_dir=job_avhxx_v6_EIV_10x_11 (new runs)
    software_tag = 'v0.3Bet' # job_dir=job_avhxx_v6_EIV_10x_11 (new runs)
    plotstem = '_'+str(ch)+'_'+software_tag+'.png'
    #--------------------------------------------------------------------------

    #
    # Load harmonisation parameters
    #

    ds = xarray.open_dataset(har_file)
    Xave = np.array(ds.parameter)
    Xcov = np.array(ds.parameter_covariance_matrix)
    Xu = np.array(ds.parameter_uncertainty) # = np.sqrt(np.diag(Xcov))
    
    #
    # Load / Generate draws
    #

    filestr_draws = "draws_" + str(ch) + "_" + str(npop) + ".npy"
    if FLAG_load_draws:
        draws = np_load(filestr_draws)
    else:
        draws = calc_draws(ds, npop)
        np_save(filestr_draws, draws, allow_pickle=False)

    #
    # Load / Generate ensemble
    #

    filestr_ensemble = "ensemble_" + str(ch) + "_" + str(npop) + ".npy"
    if FLAG_load_ensemble:
        ens = np_load(filestr_ensemble, allow_pickle=True).item()
    else:
        ens = calc_ensemble(ds, draws, nens, npop)
        np_save(filestr_ensemble, ens, allow_pickle=True)

    ensemble = ens['ensemble']
    ensemble_idx = ens['ensemble_idx']
    decile = ens['decile']
    Z = ens['Z']
    Z_norm = ens['Z_norm']

    dX = ensemble - Xave # [nens,npar]
    dXcov = np.cov(draws.T) # [npar,npar]
    dXu = np.sqrt(np.diag(dXcov)) # [npar]
    norm_u = np.linalg.norm(Xu-dXu)
    norm_cov = np.linalg.norm(Xcov-dXcov)
    print('|Xu - dXu|=',norm_u)
    print('|Xcov - dXcov|=',norm_cov)

    #
    # Open ensemble file and overwrite channel deltas and uuid:
    #

    if FLAG_write_netcdf:

        ncout = Dataset(ens_file,'r+')
        if ch == 37:
            ncout.variables['delta_params3'][:] = dX
            ncout.HARM_UUID3 = ds.uuid
        elif ch == 11:
            ncout.variables['delta_params4'][:] = dX
            ncout.HARM_UUID4 = ds.uuid
        else:
            ncout.variables['delta_params5'][:] = dX
            ncout.HARM_UUID5 = ds.uuid
        ncout.close()

    if FLAG_plot:

        plot_ensemble_closure(dX,draws,ds)
        plot_ensemble_decile_selection(Z_norm, ensemble_idx, nens)
        plot_ensemble_decile_distribution(Z, decile, npop, nens)
        plot_ensemble_deltas(dX)
        plot_ensemble_deltas_an(dX,Xu)
        plot_ensemble_deltas_normalised(dX,Xu)

    print('** END')



