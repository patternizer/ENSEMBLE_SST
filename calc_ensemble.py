#!/usr/bin/env python

# ipdb> import os; os._exit(1)

# call as: python calc_ensemble.py

# =======================================
# Version 0.1
# 28 January, 2019
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
from sklearn.preprocessing import StandardScaler
#import arpack
import scipy
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =======================================    

def calc_ensemble(file_in):

    # matchup_dataset = "AVHRR_REAL_4_RSA_____"
    # job = "job_avh11_v6_ODR_101_11.nml"
    # matchup_dataset_begin = "19911004"
    # matchup_dataset_end = "20151231"

    ds = xarray.open_dataset(file_in)

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

    X = parameter_covariance_matrix
    eig_vals, eig_vecs = np.linalg.eig(X)
    print('Eigenvectors \n%s' %eig_vecs)
    print('\nEigenvalues \n%s' %eig_vals)   

    # compute a standard eigenvalue decomposition using eigh:
    evals_all, evecs_all = eigh(X)

    # compute the largest eigenvalues (which = 'LM') of X and compare them to the known results:
    evals_large, evecs_large = eigsh(X, 3, which='LM')
    print(evals_all[-3:])
    print(evals_large)
    print(np.dot(evecs_large.T, evecs_all[:,-3:]))

    # solve for the eigenvalues with smallest magnitude:
    evals_small, evecs_small = eigsh(X, 3, which='SM', tol=1E-2)
#    evals_small, evecs_small = eigsh(X, 3, which='SM', maxiter=5000)
    print(evals_small)
    print(np.dot(evecs_small.T, evecs_all[:,:3]))

    # solve for the eigenvalues of the shift-invert mode analogous problem such that thetransformed eigenvalues will then satisfy the inverse relation where small eigenvalues become large eigenvalues:

    evals_small, evecs_small = eigsh(X, 3, sigma=0, which='LM')
    print(evals_small)
    print(np.dot(evecs_small.T, evecs_all[:,:3]))

    # to find internal eigenvalues and eigenvectors, e.g. those nearest to 1, simply set sigma = 1 and ARPACK takes care of the rest:

    evals_mid, evecs_mid = eigsh(X, 3, sigma=1, which='LM')
    i_sort = np.argsort(abs(1. / (1 - evals_all)))[-3:]
    print(evals_all[i_sort])
    print(evals_mid)
    print(np.dot(evecs_mid.T, evecs_all[:,i_sort]))

if __name__ == "__main__":

#    parser = OptionParser("usage: %prog file_in")
#    (options, args) = parser.parse_args()
#    file_in = args[0]

    file_in = "harm_FO_3.0_27b18eb_ODR_101_11_R____AVHRR_REAL_4_RSA______19911004_20151231.nc"
    calc_ensemble(file_in)


