#!/usr/bin/env python

# ipdb> import os; os._exit(1)

# call as: python calc_ensemble.py
# NB: include code: plot_ensemble.py

# =======================================
# Version 0.15
# 22 May, 2019
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
import sklearn
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

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

def find_nearest(array, value):

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()

    return array[idx], idx
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

def convert_L_BT(L, L_delta, mtac3a, mtac3b, nch):
    '''
    Look-up tables to convert radiance from the Measurement Equation to brightness temperature (BT) and vice-versa
    '''
    lut_bt_3a = mtac3a.lookup_table_BT
    lut_l_3a = mtac3a.lookup_table_radiance
    lut_bt_3b = mtac3b.lookup_table_BT
    lut_l_3b = mtac3b.lookup_table_radiance

    BT = np.empty(shape=L_delta.shape[0])
    BT_delta = np.empty(shape=(L_delta.shape[0],L_delta.shape[1]))

    if nch == 37: channel = 3
    elif nch == 11: channel = 4
    else: channel = 5

    BT = np.interp(L, lut_l_3a[:,channel], lut_bt_3a[:,channel])
    BT_delta = np.interp(L_delta, lut_l_3a[:,channel], lut_bt_3a[:,channel])  

    return BT, BT_delta

def calc_radiance_ensemble(ds, ensemble, isensor, nens, nch, fcdr):
    '''
    Calculate radiance ensemble for sensor orbit channel data
    Aim: use channel-dependent measurement equations to determine how many PCs are needed to 
    guarantee 0.001 K (1 mK) accuracy

    NB: harmonisation best-case + ensemble parameters --> a(1), a(2), a(3) and a(4) where relevant
    NB: radiance --> BT conversion is done approximately by interpolation of the FCDR LUT
    NB: radiance sensitivity is calculated using AVHRR noise characterisation data for::
         i) e_ICT (emissivity)
        ii) ccounts data: C_e, C_s and C_ict
       iii) L_ict
        iv) T_ict (normalised by T_mean and T_sdev)
    '''

    ni = fcdr.nx
    nj = fcdr.ny
    t = fcdr.time
    lat = fcdr.latitude
    lon = fcdr.longitude

    C_ict_37 = fcdr.ch3_bb_counts  
    C_s_37 = fcdr.ch3_space_counts
    C_e_37 = fcdr.ch3_earth_counts
    L_ict_37 = fcdr.ICT_Rad_Ch3    
    C_ict_11 = fcdr.ch4_bb_counts  
    C_s_11 = fcdr.ch4_space_counts 
    C_e_11 = fcdr.ch4_earth_counts 
    L_ict_11 = fcdr.ICT_Rad_Ch4    
    C_ict_12 = fcdr.ch5_bb_counts  
    C_s_12 = fcdr.ch5_space_counts 
    C_e_12 = fcdr.ch5_earth_counts 
    L_ict_12 = fcdr.ICT_Rad_Ch5   
    T_inst = fcdr.prt              

    e_ict = 0.985140
    scan_middle = int(len(ni)/2)

    gd_cold_pixel_37 = C_e_37 == C_e_37.min()
    gd_warm_pixel_37 = C_e_37 == C_e_37.max()
    gd_cold_pixel_11 = C_e_11 == C_e_11.min()
    gd_warm_pixel_11 = C_e_11 == C_e_11.max()
    gd_cold_pixel_12 = C_e_12 == C_e_12.min()
    gd_warm_pixel_12 = C_e_12 == C_e_12.max()

    cold_pixel_37 = np.where(gd_cold_pixel_37 > 0)[0][0]
    warm_pixel_37 = np.where(gd_warm_pixel_37 > 0)[0][0]
    cold_pixel_11 = np.where(gd_cold_pixel_11 > 0)[0][0]
    warm_pixel_11 = np.where(gd_warm_pixel_11 > 0)[0][0]
    cold_pixel_12 = np.where(gd_cold_pixel_12 > 0)[0][0]
    warm_pixel_12 = np.where(gd_warm_pixel_12 > 0)[0][0]

    # Ralf Quast: sensor configurations:
    #
    # | Sensor | T_min (K) | T_max (K) | T_mean (K) | T_sdev (K) | Measurement equation (11 µm & 12 µm / 3.7 µm) |
    # |--------|-----------|-----------|------------|------------|-----------------------------------------------| 
    # | m02    | 285.9     | 286.4     | 286.125823 | 0.049088   | 102 / 106                                     |
    # | n19    | 286.9     | 288.3     | 287.754638 | 0.117681   | 102 / 106                                     |
    # | n18    | 286.6     | 290.2     | 288.219774 | 0.607697   | 102 / 106                                     |
    # | n17    | 286.1     | 298.2     | 288.106630 | 1.607656   | 102 / 106                                     |
    # | n16    | 287.2     | 302.0     | 292.672201 | 3.805704   | 102 / 106                                     |
    # | n15    | 285.1     | 300.6     | 294.758564 | 2.804361   | 102 / 106                                     |
    # | n14    | 286.8     | 296.4     | 288.637636 | 1.053762   | 102 / 106                                     |
    # | n12    | 287.2     | 302.8     | 290.327113 | 2.120666   | 102 / 106                                     |
    # | n11    | 286.1     | 299.9     | 290.402168 | 3.694937   | 102 / 106                                     |
  
    T_ave = np.array([ 286.125823, 287.754638, 288.219774, 288.106630, 292.672201, 294.758564, 288.637636, 290.327113, 290.402168 ])
    T_std = np.array([ 0.049088, 0.117681, 0.607697, 1.607656, 3.805704, 2.804361, 1.053762, 2.120666, 3.694937])

    parameter = ds['parameter']
    L = np.empty(shape=len(nj),)
    L_delta = np.empty(shape=(len(nj),nens))

    if nch == 37:

        C_ict = C_ict_37
        C_s = C_s_37
        C_e = C_e_37[:,scan_middle]
        L_ict = L_ict_37

        npar = 3
        T_mean = T_ave[isensor]
        T_sdev = T_std[isensor]
        j = isensor * npar
        a0 = parameter[j]
        a1 = parameter[j+1]
        a2 = parameter[j+2]

        # Measurement equation 106:
        L = a0 + ((L_ict * (e_ict + a1)) / (C_ict - C_s)) * (C_e - C_s) + a2 * (T_inst - T_mean) / T_sdev

        for k in range(nens):
                
            b0 = ensemble[k,j]
            b1 = ensemble[k,j+1]
            b2 = ensemble[k,j+2]

            # Measurement equation 106:
            L_ens = b0 + ((L_ict * (e_ict + b1)) / (C_ict - C_s)) * (C_e - C_s) + b2 * (T_inst - T_mean) / T_sdev
            L_delta[:,k] = L_ens

        # NB: for 3.7 micron channel, scale by factor of 100 to get correct
        L = L * 100.0
        L_delta = L_delta * 100.0

    elif nch == 11:

        C_ict = C_ict_11
        C_s = C_s_11
        C_e = C_e_11[:,scan_middle]
        L_ict = L_ict_11

        npar = 4
        T_mean = T_ave[isensor]
        T_sdev = T_std[isensor]
        j = isensor * npar
        a0 = parameter[j]
        a1 = parameter[j+1]
        a2 = parameter[j+2]
        a3 = parameter[j+3]

        # Measurement equation 102:
        L = a0 + ((L_ict * (e_ict + a1)) / (C_ict - C_s) + a2 * (C_e - C_ict)) * (C_e - C_s) + a3 * (T_inst - T_mean) / T_sdev

        for k in range(nens):
                
            b0 = ensemble[k,j]
            b1 = ensemble[k,j+1]
            b2 = ensemble[k,j+2]
            b3 = ensemble[k,j+3]

            # Measurement equation 102:
            L_ens = b0 + ((L_ict * (e_ict + b1)) / (C_ict - C_s) + b2 * (C_e - C_ict)) * (C_e - C_s) + b3 * (T_inst - T_mean) / T_sdev
            L_delta[:,k] = L_ens

    else:

        C_ict = C_ict_12
        C_s = C_s_12
        C_e = C_e_12[:,scan_middle]
        L_ict = L_ict_12

        npar = 4
        T_mean = T_ave[isensor]
        T_sdev = T_std[isensor]
        j = isensor * npar
        a0 = parameter[j]
        a1 = parameter[j+1]
        a2 = parameter[j+2]
        a3 = parameter[j+3]
            
        # Measurement equation 102:
        L = a0 + ((L_ict * (e_ict + a1)) / (C_ict - C_s) + a2 * (C_e - C_ict)) * (C_e - C_s) + a3 * (T_inst - T_mean) / T_sdev

        for k in range(nens):
                
            b0 = ensemble[k,j]
            b1 = ensemble[k,j+1]
            b2 = ensemble[k,j+2]
            b3 = ensemble[k,j+3]

            # Measurement equation 102:
            L_ens = b0 + ((L_ict * (e_ict + b1)) / (C_ict - C_s) + b2 * (C_e - C_ict)) * (C_e - C_s) + b3 * (T_inst - T_mean) / T_sdev
            L_delta[:,k] = L_ens

    return L, L_delta
        
def plot_radiance_deltas(L, L_delta, nens, nch):

    fig, ax = plt.subplots()
    for k in range(nens):

        label_str = 'Ens(' + str(k+1) + ')'
        plt.plot(L - L_delta[:,k], linewidth=1.0, label=label_str)

    plt.legend(fontsize=10, ncol=1)
    ax.set_ylabel('Radiance difference', fontsize=12)
    plt.tight_layout()            
    file_str = "radiance_ensemble_" + str(nch) + ".png"
    plt.savefig(file_str)        
    plt.close('all')

def plot_bt_deltas(BT, BT_delta, nens, nch):

    fig, ax = plt.subplots()
    for k in range(nens):

        label_str = 'Ens(' + str(k+1) + ')'
        plt.plot(BT - BT_delta[:,k], linewidth=1.0, label=label_str)

    plt.legend(fontsize=10, ncol=1)
    ax.set_ylabel('BT difference', fontsize=12)
    plt.tight_layout()            
    file_str = "bt_ensemble_" + str(nch) + ".png"
    plt.savefig(file_str)        
    plt.close('all')

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

def calc_ensemble(ds, draws, sensor, nens, npop):
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

def calc_pca(ds, draws, nens):
    '''
    Apply PCA to the draw matrix
    '''

    npar = len(ds.parameter)
    X = draws
    X_ave = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)

    pca = PCA().fit(X)
    pca_var = np.cumsum(pca.explained_variance_ratio_) * 100.0

#    idx = np.where(pca_var > 0.99)
#    idx = 10-1
    idx = len(pca_var)-1

    fig = plt.figure()
    plt.plot(range(1,len(pca_var)+1), pca_var, marker='o', linestyle='-', color='r')
    plt.scatter(idx, pca_var[idx])
    plt.xticks(range(1,len(pca_var)+1))
    plt.xlabel('Number of PCs')
    plt.ylabel('Cumulative explained variance (%)')
    title_str = str(idx+1) + ' PCs account for ' + "{0:.5f}".format(pca_var[idx]) + '% of the variance'
    file_str = 'pca_variance.png'
    plt.title(title_str)
    plt.savefig(file_str)
    plt.close()

    #
    # Preserve 90% of the variance with PCA
    #

    # n_PC = pca.n_components_
    # n_PC = idx[0][0]
    n_PC = idx

    # PC = pca.transform(X)
    # Xhat = pca.inverse_transform(PC)

    Xhat = np.dot(pca.transform(X)[:,:n_PC], pca.components_[:n_PC,:])
    Xhat += X_ave
    
    # NB: if X is a correlation matrix we must first multiply by X_std then add X_ave

    for i in range(0,npar):

        fig = plt.figure()
        plt.plot(X[:,i],'blue')
        plt.plot(Xhat[:,i],'red')
        plt.xlabel('Draw')
        plt.ylabel('Parameter value')
        title_str = 'Harmonisation parameter: ' + str(i+1)
        file_str = 'pca_fit' + str(i+1) + '.png'
        plt.title(title_str)
        plt.savefig(file_str)
        plt.close()

# =======================================    
# INCLUDE PLOT CODE:
exec(open('plot_ensemble.py').read())
# =======================================    

# =======================================    
# MAIN BLOCK
# =======================================    
    
if __name__ == "__main__":

    #--------------------------------------------------
#    parser = OptionParser("usage: %prog nch npop nens")
#    (options, args) = parser.parse_args()

#    nch = args[0]
#    npop = args[1]
#    nens = args[2]

    nch = 37
#    nch = 11
#    nch = 12
    npop = 1000000
    nens = 11
    #--------------------------------------------------
    
    file_in = "FIDUCEO_Harmonisation_Data_" + str(nch) + ".nc"
    filestr_draws = "draws_" + str(nch) + "_" + str(npop) + ".npy"
    filestr_ensemble = "ensemble_" + str(nch) + "_" + str(npop) + ".npy"
    filestr_ensemble_idx = "ensemble_idx_" + str(nch) + "_" + str(npop) + ".npy"

    sensor = ['METOPA','NOAA19','NOAA18','NOAA17','NOAA16','NOAA15','NOAA14','NOAA12','NOAA11']

    #
    # Runtime flags
    #

    flag_load_draws = 1
    flag_load_ensemble = 1
    flag_pca = 0
    flag_export = 0
    flag_plot = 1

    #
    # Load L1B orbit data
    #
    fcdr = xarray.open_dataset("avhrr_fcdr_full.nc", decode_times=False)

    #
    # Load radiance and BT look-up tables
    #
    mtac3a = xarray.open_dataset("FIDUCEO_FCDR_L1C_AVHRR_MTAC3A_20110619225807_20110620005518_EASY_v0.2Bet_fv2.0.0.nc")
    mtac3b = xarray.open_dataset("FIDUCEO_FCDR_L1C_AVHRR_MTAC3B_20110619231357_20110620013100_EASY_v0.2Bet_fv2.0.0.nc")

    ds = load_data(file_in)

    if flag_load_draws:

        draws = np_load(filestr_draws)

    else:

        draws = calc_draws(ds, npop)
        np_save(filestr_draws, draws, allow_pickle=False)

    if flag_load_ensemble:

        ensemble = np_load(filestr_ensemble)
        ensemble_idx = np_load(filestr_ensemble_idx)

    else:

        if flag_pca:

            ensemble, ensemble_idx = calc_pca(ds, draws, nens)

        else:

            ensemble, ensemble_idx = calc_ensemble(ds, draws, sensor, nens, npop)

        np_save(filestr_ensemble, ensemble, allow_pickle=False)
        np_save(filestr_ensemble_idx, ensemble_idx, allow_pickle=False)

    isensor = 0
    L, L_delta = calc_radiance_ensemble(ds, ensemble, isensor, nens, nch, fcdr)
    BT, BT_delta = convert_L_BT(L, L_delta, mtac3a, mtac3b, nch)

    if flag_export:

        export_ensemble(ensemble)

    if flag_plot:

        plot_radiance_deltas(L, L_delta, nens, nch)
        plot_bt_deltas(BT, BT_delta, nens, nch)

        plot_ensemble_check(ds, ensemble)
        plot_ensemble_deltas(ds, ensemble, sensor, nens)
        plot_bestcase_parameters(ds, sensor)
        plot_bestcase_covariance(ds)
        plot_population_coefficients(ds, draws, sensor, npop)
        plot_population_histograms(ds, draws, sensor, nens)
        plot_population_cdf(ds, draws, sensor, nens, npop)

