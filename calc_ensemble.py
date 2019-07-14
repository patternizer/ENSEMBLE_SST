#!/usr/bin/env python

# ipdb> import os; os._exit(1)

# call as: python calc_ensemble.py
# NB: include code: plot_ensemble.py

# =======================================
# Version 0.24
# 25 June, 2019
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
from netCDF4 import Dataset
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
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import sklearn
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import convert_func as con

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

def read_in_LUT(avhrr_sat):
    '''
    Load radiance and BT look-up tables for all sensors in avhrr_sat
    '''

    LUT = {}
    all_lut_radiance_dict = np.load('lut_radiance.npy', encoding='bytes').item()
    all_lut_BT_dict = np.load('lut_BT.npy', encoding='bytes').item()

    try:
        LUT['L'] = all_lut_radiance_dict[avhrr_sat][:]
        LUT['BT'] = all_lut_BT_dict[avhrr_sat][:]
    except:
        print("Sensor for AVHRR does not exist: ", avhrr_sat)

    return LUT

def rad2bt(L, channel, lut):
    '''
    Convert radiance L to BT using LUT for channel
    '''

    BT = np.interp(L,lut['L'][:,channel], lut['BT'][:,channel], left=-999.9,right=-999.9)

    return BT

def bt2rad(bt, channel, lut):
    '''
    Convert BT to radiance L using LUT for channel
    '''

    L = np.interp(BT,lut['BT'][:,channel], lut['L'][:,channel], left=-999.9,right=-999.9)

    return L

def count2rad(Ce, Cs, Cict, Lict, Tinst, channel, a0, a1, a2, a3):
    '''
    Calculate radiance from scan counts and temperatures for channel using harmonisation coefficients
    '''

    L = np.empty(shape=(Ce.shape[0], Ce.shape[1]))

    try:

        if channel == 3:
            L = a0 + ((Lict * (0.985140 + a1)) / (Cict - Cs)) * (Ce - Cs) + a2 * Tinst

        elif channel > 3:
            L = a0 + ((Lict * (0.985140 + a1)) / (Cict - Cs) + a2 * (Ce - Cict)) * (Ce - Cs) + a3 * Tinst
    except:

        print("No FIDUCEO thermal channel selected: channel=", channel, " < 3")

    return L

# =======================================    

def load_data(file_in):
    '''
    Load harmonisation parameters and covariance matrix
    '''
    ds = xarray.open_dataset(file_in)
    return ds

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

    return draws

def calc_ensemble_draws(ds, draws, nens, npop):
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

        # --------------------------------
        # NUMPY: sort --> unsort example
        # --------------------------------
        # arr = np.array([5, 3, 7, 2, 46, 3]) 
        # arr_sorted = np.sort(arr)
        # arr_idx = np.argsort(arr)
        # arr_remade = np.empty(arr.shape)
        # arr_remade[arr_idx] = arr_sorted
        # --------------------------------
  
        #
        # Decile values of Z_cdf (for each parameter)
        # 

        idx_step = int(npop / (nens - 1))
        # deciles = np.linspace(0, nens, nens+1, endpoint=True).astype('int') * 10

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

        # y = np.percentile(Z_norm, deciles[j+1], interpolation='nearest') 
        # i = abs(Z_norm - y).argmin()        
        
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
        
def calc_pca(ds, draws, nens):
    '''
    Compare eigenvalue decomposition (EVD) of the Harmonisation covariance matrix with
    singular value decomposition (SVD) of the draw matrix of varying size.
    '''

    #----------------------------------------------------------------------------
    # PCA
    #----------------------------------------------------------------------------
    # z1 = l11 * x1 + l12 * x2 + l13 * x3 + ... + l1p * xp --> this is the 1st PC
    # z2 = l21 * x1 + l22 * x2 + l23 * x3 + ... + l2p * xp
    # z3 = l31 * x1 + l32 * x2 + l33 * x3 + ... + l3p * xp
    # ...
    # zm = lm1 * x1 + lm2 * x2 + lm3 * x3 + ... + lmp * xp
    #----------------------------------------------------------------------------
    # li1^2 + li2^2 + li3^2 + ... lip^2 = 1 --> this is the 1st eigenvector
    #----------------------------------------------------------------------------
    
    #
    # EVD of Harmonisation covariance matrix (H = Y_cov)
    #
    
    U = np.array(ds.parameter_uncertainty)
    R = np.array(ds.parameter_correlation_matrix) 
    C = np.array(ds.parameter_covariance_matrix)
    Y = np.array(ds.parameter)

    X = R

    nparameter = len(Y)

    eigen_val, eigen_vec = calc_eigen(X)
    L = eigen_val
    V = eigen_vec
    S = np.diag(np.sqrt(L))
    T = np.matmul(V, R)

    #
    # SVD reconstruction
    #

    svd_U, svd_S, svd_V = calc_svd(X)
    Z_SVD = np.matmul(np.matmul(svd_U[:, :n_PC], np.diag(svd_S[:n_PC])),svd_V[:n_PC, :])






    #
    # Range of draw population sizes: len(parameter) * 2^n; n=0:10
    #

    ndraws = nparameter * 2**np.arange(10, dtype = np.uint64)[::1]

    # Plot relative variance explained [%) for EVD versus SVD (draws)

    S_val = np.empty(shape=(nparameter, len(ndraws)))

    fig = plt.figure()
    for i in range(len(ndraws)):
        
        size = ndraws[i]
        draws = np.random.multivariate_normal(Y, C, size)
        svd_U, svd_S, svd_V = calc_svd(draws - draws.mean(axis=0))        
        S_val[:,i] = svd_S**2 / sum(svd_S**2)

        label_str = 'SVD: n(draws)=' + str(ndraws[i])
        plt.plot(range(1,nparameter+1), 100 * S_val[:,i], linewidth=1.0, label=label_str)

    plt.plot(range(1,nparameter+1), 100 * eigen_val / sum(eigen_val), 'k.', linestyle='--', linewidth=0.5, label='Eigenvalue decompositition of cov(H)')
    plt.xticks(range(1,nparameter+1))
    plt.xlim([0.5,20.5])
    plt.legend(fontsize=8, ncol=1)
    plt.xlabel('Number of PCs')
    plt.ylabel('Relative variance explained: EVD(H) versus SVD(draws) [%]')
    file_str = 'pca_svd_versus_evd1.png'
    plt.savefig(file_str)
    plt.close()

    fig = plt.figure()
    for i in range(len(ndraws)):

        label_str = 'SVD: n(draws)=' + str(ndraws[i])
        plt.plot(range(1,nparameter+1), 100 * eigen_val / sum(eigen_val) - 100 * S_val[:,i], linewidth=1.0, label=label_str)

    plt.xticks(range(1,nparameter+1))
    plt.xlim([0.5,20.5])
    plt.legend(fontsize=8, ncol=1)
    plt.xlabel('Number of PCs')
    plt.ylabel('Relative variance explained: EVD(H) - SVD(draws) [%] ')
    file_str = 'pca_svd_versus_evd2.png'
    plt.savefig(file_str)
    plt.close()

def calc_ensemble_pca(ds, draws, nens):
    '''
    Apply PCA to the draw matrix
    '''

#    U = np.diag(ds.parameter_uncertainty)
    U = np.array(ds.parameter_uncertainty)
    R = np.array(ds.parameter_correlation_matrix) 
#   C = np.matmul(np.matmul(U,R),U)
    C = np.array(ds.parameter_covariance_matrix)
    Y = np.array(ds.parameter)

    X = R
    Z = draws

    nparameter = len(Y)

    ensemble = np.empty(shape=(nens, nparameter))
    ensemble_idx = np.empty(shape=(nens, nparameter))

    n_PC = nparameter

    #
    # EVD decomposition
    #

    eigen_val, eigen_vec = calc_eigen(X)
    L = eigen_val
    V = eigen_vec
    S = np.diag(np.sqrt(L))
    T = np.matmul(V, R)

    #
    # SVD reconstruction
    #

    svd_U, svd_S, svd_V = calc_svd(X)
    Z_SVD = np.matmul(np.matmul(svd_U[:, :n_PC], np.diag(svd_S[:n_PC])),svd_V[:n_PC, :])

    for n_PC in range(1,nparameter+1):

        Z_SVD = np.matmul(np.matmul(svd_U[:, :n_PC], np.diag(svd_S[:n_PC])),svd_V[:n_PC, :])
#        fig, ax = plt.subplots()
#        vmin, vmax = -1.0, 1.0
#        sns.heatmap(Z_SVD, center=(vmin+vmax)/2.0, vmin=vmin, vmax=vmax)
#        titlestr = 'X (SVD): n_PC = ' + str(n_PC)
#        filestr = 'SVD_X_n_PC_' + str(n_PC) + '.png'
#        plt.title(titlestr)
#        plt.savefig(filestr)
#        plt.close()

#        fig, ax = plt.subplots()
#        vmin, vmax = -1.0, 1.0
#        sns.heatmap( X - Z_SVD, center=(vmin+vmax)/2.0, vmin=vmin, vmax=vmax)
#        titlestr = 'dX (SVD): n_PC = ' + str(n_PC)
#        filestr = 'SVD_dX_n_PC_' + str(n_PC) + '.png'
#        plt.title(titlestr)
#        plt.savefig(filestr)
#        plt.close()

    #
    # PCA reconstruction
    #

    X = Z
    pca = PCA().fit(X)
#    Z_PCA = np.dot(pca.transform(X)[:,:n_PC], pca.components_[:n_PC,:]) + pca.mean_[:n_PC]
#    Z_PCA = np.matmul(pca.transform(X)[:,:n_PC], pca.components_[:n_PC,:]) + pca.mean_[:n_PC]

    fig, ax = plt.subplots()
    for n_PC in range(1,nparameter+1):

        Z_PCA = []
        Z_PCA = np.matmul(pca.transform(X)[:,:n_PC], pca.components_[:n_PC,:]) + pca.mean_
        labelstr = 'n_PC=' + str(n_PC)
        plt.plot( ((Y - Z_PCA.mean(axis=0))/Y), label=labelstr)

    plt.legend(fontsize=8, ncol=2)
    titlestr = 'parameter delta, dY(n_PC)'
    filestr = 'PCA_dY_n_PC.png'
    plt.title(titlestr)
    plt.savefig(filestr)
    plt.close()

    fig, ax = plt.subplots()
    for n_PC in range(1,nparameter+1):

        Z_PCA = []
        Z_PCA = np.matmul(pca.transform(X)[:,:n_PC], pca.components_[:n_PC,:]) + pca.mean_
        labelstr = 'n_PC=' + str(n_PC)
        plt.plot( ((U - Z_PCA.std(axis=0))/U), label=labelstr)

    plt.legend(fontsize=8, ncol=2)
    titlestr = 'uncertainty delta, dU(n_PC)'
    filestr = 'PCA_dU_n_PC.png'
    plt.title(titlestr)
    plt.savefig(filestr)
    plt.close()

    ############################
    # TEST:
    # 1) set Z=draws
    # 2) set n_PC = 1
    # 3) take 10 draws of Z_PCA
    # 4) repeat for other PC axes
    # 5) reconstruct ensemble
    ############################

    # NB: if X is a correlation matrix we must first multiply by std then add mean

    # pca_cumvar = np.cumsum(pca.explained_variance_ratio_) * 100.0

#    isensor = 0
#    fcdr = xarray.open_dataset("avhrr_fcdr_full.nc", decode_times=False) 
#    lut = xarray.open_dataset("FIDUCEO_FCDR_L1C_AVHRR_MTAC3A_20110619225807_20110620005518_EASY_v0.2Bet_fv2.0.0.nc")
#    L, L_delta = calc_L_ensemble(ds, ensemble, isensor, nens, nch, fcdr)
#    BT, BT_delta = calc_BT_ensemble(L, L_delta, nch, lut)

    # Calc radiance of best-case 
    # Calc radiance of Xhat
    # Calc BT of best-case
    # Calc BT of Xhat
    # Cal BT diff
    # if BT diff < 0.001K output n_PC; stop
    # Sample eigenvectors --> 10-member ensemble
    # export ensemble

    cov_par = C
    cov_ensemble = np.cov(ensemble, rowvar=False)
    cov_diff = cov_par - cov_ensemble
    corr_par = R
    corr_ensemble = np.corrcoef(ensemble, rowvar=False)
    corr_diff = corr_par - corr_ensemble

    return ensemble, ensemble_idx

# =======================================    
# INCLUDE PLOT CODE:
exec(open('plot_ensemble.py').read())
# =======================================    

# =======================================    
# MAIN BLOCK
# =======================================    
    
if __name__ == "__main__":

    #--------------------------------------------------
    # parser = OptionParser("usage: %prog ch npop nens idx l1b_file harm_file")
    # (options, args) = parser.parse_args()

    # ch = args[0]
    # npop = args[1]
    # nens = args[2]
    # idx = args[3]
    # l1b_file = args[4]
    # harm_file = args[5]

    ################################################################
    # RUN PARAMETERS:
    ################################################################

    ch = 37
    npop = 1000000
    nens = 11
    idx = 7          # MTA (see avhrr_sat)
    l1b_file = 'mta_mmd.nc'
    harm_file = 'FIDUCEO_Harmonisation_Data_' + str(ch) + '.nc'

    flag_load_draws = 1
    flag_load_ensemble = 1 
    flag_load_L_BT = 1
    flag_pca = 0
    flag_plot = 0
    flag_plot_L_BT = 1

    if idx == 7:
        noT = True
    else:
        noT = False
    if ch == 37:
        channel = 3
    elif ch == 11:
        channel = 4
    else:
        channel = 5

    #
    # Load harmonisation parameters
    #

    ds = xarray.open_dataset(harm_file)

    flag_new = False # NEW harmonisation structure (run >= '3.0-4d111a1')

    sensor = ['METOPA','NOAA19','NOAA18','NOAA17','NOAA16','NOAA15','NOAA14','NOAA12','NOAA11']

    #
    # Load / Generate draws
    #

    filestr_draws = "draws_" + str(ch) + "_" + str(npop) + ".npy"

    if flag_load_draws:

        draws = np_load(filestr_draws)

    else:

        draws = calc_draws(ds, npop)
        np_save(filestr_draws, draws, allow_pickle=False)

    #
    # Load / Generate ensemble
    #

    filestr_ensemble = "ensemble_" + str(ch) + "_" + str(npop) + ".npy"
    filestr_ensemble_idx = "ensemble_idx_" + str(ch) + "_" + str(npop) + ".npy"

    if flag_load_ensemble:

        ensemble = np_load(filestr_ensemble)
        ensemble_idx = np_load(filestr_ensemble_idx)

    else:

        if flag_pca:

            calc_pca(ds, draws, nens)
            ensemble, ensemble_idx = calc_ensemble_pca(ds, draws, nens)

        else:

            ensemble, ensemble_idx = calc_ensemble_draws(ds, draws, nens, npop)
 
        # np_save(filestr_ensemble, ensemble, allow_pickle=False)
        # np_save(filestr_ensemble_idx, ensemble_idx, allow_pickle=False)

    if flag_plot:

        plot_bestcase_parameters(ds, sensor)
        plot_bestcase_covariance(ds)
        plot_population_coefficients(ds, draws, sensor, npop)
        plot_population_histograms(ds, draws, sensor, nens)
        plot_population_cdf(ds, draws, sensor, nens, npop)
        plot_ensemble_deltas(ds, ensemble, sensor, nens)
        plot_ensemble_check(ds, ensemble)
        
    if flag_load_L_BT:

        filestr_orbit_lat = "orbit_lat" + ".npy"
        filestr_orbit_lon = "orbit_lon" + ".npy"
        filestr_L = "L_" + str(ch) + "_" + str(npop) + ".npy"
        filestr_BT = "BT_" + str(ch) + "_" + str(npop) + ".npy"
        filestr_L_ensemble = "L_ensemble_" + str(ch) + "_" + str(npop) + ".npy"
        filestr_BT_ensemble = "BT_ensemble_" + str(ch) + "_" + str(npop) + ".npy"
        lat = np_load(filestr_orbit_lat)
        lon = np_load(filestr_orbit_lon)
        L = np_load(filestr_L)
        BT = np_load(filestr_BT)
        L_delta = np_load(filestr_L_ensemble)
        BT_delta = np_load(filestr_BT_ensemble)

    else:

        ################################################################
        # SENSOR-SPECIFIC RUN:
        ################################################################
        
        idx = 7      # MTA (see avhrr_sat)
        l1b_file = 'mta_l1b.nc'            
        avhrr_sat = [b'N12',b'N14',b'N15',b'N16',b'N17',b'N18',b'N19',b'MTA',b'MTB']
        # NB: Harmonisation coefficients provided by Ralf Quast are for MTA --> N11:
        # RQ:          MTA,   N19,   N18,   N17,   N16,   N15,  N14,    N12,   N11
        # index         0       1      2      3      4      5      6     7      8
        # --> new index map

        idx_ = 7 - idx

        T_mean = np.array([ 290.327113, 288.637636, 294.758564, 292.672201, 288.106630, 288.219774, 287.754638, 286.125823, np.nan])
        T_sdev = np.array([ 2.120666, 1.053762, 2.804361, 3.805704, 1.607656, 0.607697, 0.117681, 0.049088, np.nan])

        #
        # Load: lut
        #

        lut = read_in_LUT(avhrr_sat[idx])

        #
        # Load: L1b orbit counts and temperatures
        #

        fcdr = xarray.open_dataset(l1b_file, decode_times=False)
        if channel == 3:
            Ce = fcdr.ch3_earth_counts
            Cs = fcdr.ch3_space_counts
            Cict = fcdr.ch3_bb_counts
            Lict = fcdr.ICT_Rad_Ch3
        elif channel == 4:
            Ce = fcdr.ch4_earth_counts
            Cs = fcdr.ch4_space_counts
            Cict = fcdr.ch4_bb_counts
            Lict = fcdr.ICT_Rad_Ch4
        else:
            Ce = fcdr.ch5_earth_counts
            Cs = fcdr.ch5_space_counts
            Cict = fcdr.ch5_bb_counts
            Lict = fcdr.ICT_Rad_Ch5
        Tinst = (fcdr.prt - T_mean[idx]) / T_sdev[idx]
        lat = fcdr.latitude
        lon = fcdr.longitude

        #
        # Load: harmonisation coefficients
        #

        if channel == 3:
            a0 = ds.parameter[(idx_ *3)].values
            a1 = ds.parameter[(idx_ *3)+1].values
            a2 = ds.parameter[(idx_ *3)+2].values
            a3 = np.nan
        else:
            a0 = ds.parameter[(idx_ *4)].values
            a1 = ds.parameter[(idx_ *4)+1].values
            a2 = ds.parameter[(idx_ *4)+2].values
            a3 = ds.parameter[(idx_ *4)+3].values

        #
        # Calculate L from counts and temperatures with measurement equation and convert to BT
        #

        L = count2rad(Ce, Cs, Cict, Lict, Tinst, channel, a0, a1, a2, a3)
        BT = rad2bt(L, channel, lut)

        #
        # Save orbital L and BT:
        #

        filestr_orbit_lat = "orbit_lat" + ".npy"
        filestr_orbit_lon = "orbit_lon" + ".npy"
        filestr_L = "L_" + str(ch) + "_" + str(npop) + ".npy"
        filestr_BT = "BT_" + str(ch) + "_" + str(npop) + ".npy"
        np_save(filestr_orbit_lat, lat, allow_pickle=False)
        np_save(filestr_orbit_lon, lon, allow_pickle=False)
        np_save(filestr_L, L, allow_pickle=False)
        np_save(filestr_BT, BT, allow_pickle=False)
    
        #
        # Calculate ensemble radiance and ensemble BT
        #

        b3_nan = np.ones(shape=(nens,len(ds.parameter)))*np.nan    
        if channel == 3:
            b0 = ensemble[:,(idx_ *3)]
            b1 = ensemble[:,(idx_ *3)+1]
            b2 = ensemble[:,(idx_ *3)+2]
            b3 = b3_nan[:,(idx_ *3)+2]
        else:
            b0 = ensemble[:,(idx_ *4)]
            b1 = ensemble[:,(idx_ *4)+1]
            b2 = ensemble[:,(idx_ *4)+2]
            b3 = b3_nan[:,(idx_ *4)+2]
            
        L_delta = np.empty(shape=(Ce.shape[0], Ce.shape[1], nens))    
        BT_delta = np.empty(shape=(Ce.shape[0], Ce.shape[1] ,nens))    

        for k in range(nens):

            a0 = b0[k]
            a1 = b1[k]
            a2 = b2[k]
            a3 = b3[k]

            L_ens  = count2rad(Ce, Cs, Cict, Lict, Tinst, channel, a0, a1, a2, a3)
            L_delta[:,:,k] = L_ens

        BT_delta = rad2bt(L_delta, channel, lut)    
    
        #
        # Save orbital L and BT ensembles:
        #

        filestr_L_ensemble = "L_ensemble_" + str(ch) + "_" + str(npop) + ".npy"
        filestr_BT_ensemble = "BT_ensemble_" + str(ch) + "_" + str(npop) + ".npy"
        np_save(filestr_L_ensemble, L_delta, allow_pickle=False)
        np_save(filestr_BT_ensemble, BT_delta, allow_pickle=False)

        ################################################################

    if flag_plot_L_BT:

        #
        # Plot L, BT, L deltas and BT deltas along track
        #

        along_track = int(L_delta.shape[1]/2)
        bad_data = np.less_equal(BT[:,along_track],np.zeros(BT[:,along_track].shape))
        BT[bad_data, along_track] = np.nan

        plot_L_BT(L[:,along_track], BT[:,along_track], ch)
        plot_L_deltas(L[:,along_track], L_delta[:,along_track,:], nens, ch)
        plot_BT_deltas(BT[:,along_track], BT_delta[:,along_track,:], nens, ch)

        #
        # Plot orbital L and BT deltas (full scan)
        #

        # projection = 'platecarree'
        projection = 'mollweide'
        # projection = 'robinson'

        for i in range(nens):

            varstr_L = "L-L(ens[" + str(i) + "])"
            varstr_BT = "BT-BT(ens[" + str(i) + "])"
            filestr_L = "plot_orbit_L_delta_" + str(i) + ".png"
            filestr_BT = "plot_orbit_BT_delta_" + str(i) + ".png"
            titlestr_L = "L_delta: ensemble member=" + str(i)
            titlestr_BT = "BT_delta: ensemble member=" + str(i)
            bad_data = np.less_equal(L_delta[:,:,i],np.zeros(L.shape))
            L[bad_data] = np.nan
            L_delta[bad_data,i] = np.nan
            bad_data = np.less_equal(BT_delta[:,:,i],np.zeros(BT.shape))
            BT[bad_data] = np.nan
            BT_delta[bad_data,i] = np.nan            

            vmin = -0.0003
            vmax = 0.0003
            plot_orbit_var(lat, lon, np.array(L-L_delta[:,:,i]), vmin, vmax, projection, filestr_L, titlestr_L, varstr_L)
            vmin = -0.3
            vmax = 0.3
            plot_orbit_var(lat, lon, np.array(BT-BT_delta[:,:,i]), vmin, vmax, projection, filestr_BT, titlestr_BT, varstr_BT)
        
    print('** END')




