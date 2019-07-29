#!/usr/bin/env python

# call as: python cov2ensemble.py 
# include plot code: plot_cov2ensemble.py
  
# =======================================
# Version 0.16
# 28 July, 2019
# https://patternizer.github.io/
# michael.taylor AT reading DOT ac DOT uk
# =======================================

import os
import os.path
from os import fsync, remove
import sys
import glob
import optparse
from  optparse import OptionParser
from netCDF4 import Dataset
import numpy as np
from numpy import array_equal, savetxt, loadtxt, frombuffer, save as np_save, load as np_load, savez_compressed, array
from pandas import Series, DataFrame, Panel
import xarray
import scipy.linalg as la
from scipy.special import erfinv
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.datasets import make_classification
import seaborn as sns; sns.set(style='darkgrid')
import matplotlib.pyplot as plt; plt.close('all')
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.collections import PolyCollection
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

#----------------------------------------------------------------------------------
import ensemble_func as ensemble_func # functions for ensemble generation
import convert_func as convert_func # functions for L<-->BT conversion & HAR + CCI Meas Eqn
#----------------------------------------------------------------------------------

# =======================================
# INCLUDE PLOT CODE:
exec(open('plot_ensemble.py').read())
exec(open('plot_cov2ensemble.py').read())
# =======================================

if __name__ == "__main__":

    #------------------------------------------------------------------------------
    # parser = OptionParser("usage: %prog ch npop nens har_file")
    # (options, args) = parser.parse_args()
    #------------------------------------------------------------------------------
    # ch = args[0]
    # npop = args[1]
    # nens_file = args[2]
    # har_file = args[3]
    #------------------------------------------------------------------------------

    #------------------------------------------------------------------------------
    # RUN PARAMETERS
    #------------------------------------------------------------------------------
    ch = 37
    npop = 1000000
    nens = 10

    # /gws/nopw/j04/fiduceo/Users/jmittaz/FCDR/Mike/FCDR_AVHRR/GBCS/dat_cci/
    har_file = 'FIDUCEO_Harmonisation_Data_' + str(ch) + '.nc'    

    n = int(nens/2) # --> (2*n) = nens = number of ensemble members
    c = 0.99 # variance_threshold

    FLAG_load_mc = 0          # 0-->run, 1-->load
    FLAG_mns = 1              # 0-->CRS, 1-->MNS
    FLAG_load_draws = 1       
    FLAG_load_ensemble = 1    
    FLAG_write_netcdf = 0     
    FLAG_plot = 1

    # software_tag = '3e8c463' # job dir=job_avhxx_v6_EIV_10x_11 (old runs)
    # software_tag = '4d111a1' # job_dir=job_avhxx_v6_EIV_10x_11 (new runs)
    software_tag = 'v0.3Bet' # job_dir=job_avhxx_v6_EIV_10x_11 (new runs)
    #--------------------------------------------------------------------------
    
    senspr = [b'N12',b'N14',b'N15',b'N16',b'N17',b'N18',b'N19',b'MTA',b'MTB']

    if FLAG_mns:
        plotstem = '_'+str(ch)+'_'+'MNS'+'_'+software_tag+'.png'
        mc_file = 'MC_Harmonisation_MNS.nc'
    else:
        plotstem = '_'+str(ch)+'_'+'CRS'+'_'+software_tag+'.png'
        mc_file = 'MC_Harmonisation_CRS.nc'

    #
    # Load harmonisation parameters
    #

    har = xarray.open_dataset(har_file, decode_cf=True)
    a_ave = np.array(har.parameter)
    a_cov = np.array(har.parameter_covariance_matrix)
    a_u = np.array(har.parameter_uncertainty)

    if FLAG_load_mc:

        # Load ensemble

        ncout = Dataset(mc_file,'r')
        if ch == 37:
            da = ncout.variables['delta_params3'][:]
        elif ch == 11:
            da = ncout.variables['delta_params4'][:]
        else:
            da = ncout.variables['delta_params5'][:]
        ncout.close()

    if FLAG_mns:

        filestr_draws = "draws_" + str(ch) + "_" + str(npop) + ".npy"
        filestr_ensemble = "ensemble_" + str(ch) + "_" + str(npop) + ".npy"

        # Load / Generate draws

        if FLAG_load_draws:
            draws = np_load(filestr_draws)
        else:
            draws = ensemble_func.cov2draws(har,npop)
            np_save(filestr_draws, draws, allow_pickle=False)

        # Load / Generate ensemble

        if FLAG_load_ensemble:
            ens = np_load(filestr_ensemble, allow_pickle=True).item()
        else:
            ens = ensemble_func.draws2ensemble(draws,nens)
            np_save(filestr_ensemble, ens, allow_pickle=True)

        ensemble = ens['ensemble']
        ensemble_idx = ens['ensemble_idx']
        decile = ens['decile']
        Z = ens['Z']
        Z_norm = ens['Z_norm']
        da = ensemble - a_ave # [nens,npar]

    else:

        ev = ensemble_func.cov2ev(a_cov,c)
        da = ensemble_func.ev2da(ev,n) # sum of nPC (99% variance) 
        
        # Projection onto first 2 PCs:

        da_pc12 = ensemble_func.da2pc12(ev,n)
        plot_ensemble_deltas_pc12(da_pc12,a_u)

    # Ensemble closure

    da_cov = np.cov(da.T) # [npar,npar]
    da_u = np.sqrt(np.diag(da_cov)) # [npar]
    norm_u = np.linalg.norm(a_u - da_u)
    norm_cov = np.linalg.norm(a_cov - da_cov)
    print('|a_u - da_u|=',norm_u)
    print('|a_cov - da_cov|=',norm_cov)

    # Open ensemble file and overwrite channel deltas and uuid:

    if FLAG_write_netcdf:

        ncout = Dataset(mc_file,'r+')
        if ch == 37:
            ncout.variables['delta_params3'][:] = da
            ncout.HARM_UUID3 = har.uuid
        elif ch == 11:
            ncout.variables['delta_params4'][:] = da
            ncout.HARM_UUID4 = har.uuid
        else:
            ncout.variables['delta_params5'][:] = da
            ncout.HARM_UUID5 = har.uuid
        ncout.close()

    if FLAG_plot:

        if FLAG_mns:
            plot_ensemble_closure(da,draws,har) 
            plot_ensemble_decile_selection(Z_norm, ensemble_idx, nens) 
            plot_ensemble_decile_distribution(Z, decile, npop, nens)
        else:
            plot_eigenspectrum(ev)
        plot_ensemble_deltas(da)
        plot_ensemble_deltas_an(da,a_u)
        plot_ensemble_deltas_normalised(da,a_u)
        plot_crs()

        plot_bestcase_covariance(har)
        plot_bestcase_parameters(har, sensor)
        plot_ensemble_deltas(har, ensemble, sensor, nens)
        plot_population_cdf(har, draws, sensor, nens, npop)
        plot_population_coefficients(har, draws, sensor, npop)
        plot_population_histograms(har, draws, sensor, nens)

    #------------------------------------------------        
    print('** end')






