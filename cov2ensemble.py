#!/usr/bin/env python

# call as: python cov2ensemble.py 
# include plot code: plot_cov2ensemble.py
  
# =======================================
# Version 0.15
# 25 July, 2019
# https://patternizer.github.io/
# michael.taylor AT reading DOT ac DOT uk
# =======================================

from  optparse import OptionParser
import numpy as np
from numpy import array_equal, savetxt, loadtxt, frombuffer, save as np_save, load as np_load, savez_compressed, array
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
import convert_func as con   # measurement equations & L<-->BT conversion
#------------------------------------------------------------------------------

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

def pc_plot(PC,E,labels=None):
    '''
    Project data on first two PCs and draw eigenvectors
    '''

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

    # U,S,V = np.linalg.svd(X, full_matrices=True) 
    eigenvalues,eigenvectors = np.linalg.eig(X)
    eigenvalues_cumsum = (eigenvalues/eigenvalues.sum()).cumsum()
    nPC = np.where(eigenvalues_cumsum > c)[0][0] # NB: python indexing
    nPC_variance = eigenvalues_cumsum[nPC]

    print('nPC=',(nPC+1))
    print('nPC_variance=',nPC_variance)

    ev = {}
    ev['eigenvectors'] = eigenvectors
    ev['eigenvalues'] = eigenvalues
    ev['eigenvalues_sum'] = eigenvalues.sum()
    ev['eigenvalues_norm'] = eigenvalues/eigenvalues.sum()
    ev['eigenvalues_cumsum'] = eigenvalues_cumsum
    ev['eigenvalues_prod'] = eigenvalues.prod() # should be ~ det(Xcov)
    ev['eigenvalues_rank'] = np.arange(1,len(eigenvalues)+1) # for plotting
    ev['nPC'] = nPC
    ev['nPC_variance'] = nPC_variance

    return ev

def ev2da(n,ev):
    '''
    Create (2*n) deltas of Xave using (un)constrained random sampling
    '''
    nPC = ev['nPC']
    eigenvalues = ev['eigenvalues']
    eigenvectors = ev['eigenvectors']
    nparameters = eigenvectors.shape[1]
    da = np.zeros(shape=(2*n,nparameters))
    for k in range(nPC):
        randomized = np.sort(np.array(crs.generate_10(n)))
        da_c = np.zeros(shape=(2*n,nparameters))
        for i in range((2*n)):        
            da_c[i,:] = randomized[i] * np.sqrt(eigenvalues[k]) * eigenvectors[k,:]
        da = da + da_c

    return da

def calc_da_pc12(n,ev):
    '''
    Create (2*n) deltas of Xave using (un)constrained random sampling
    '''
    nPC = ev['nPC']
    eigenvalues = ev['eigenvalues']
    eigenvectors = ev['eigenvectors']
    nparameters = eigenvectors.shape[1]
    # randomized = np.sort(np.array(crs.generate_10_single(n))) # unconstrained
    randomized = np.sort(np.array(crs.generate_10(n)))          # constrained
    da_pc1 = []
    da_pc2 = []
    for i in range((2*n)):        
        da_pc1.append(randomized[i] * np.sqrt(eigenvalues[0]) * eigenvectors[:,0])
        da_pc2.append(randomized[i] * np.sqrt(eigenvalues[1]) * eigenvectors[:,1])
    da_pc12 = {}
    da_pc12['da_pc1'] = np.array(da_pc1)
    da_pc12['da_pc2'] = np.array(da_pc2)

    return da_pc12

def cov2draws(ds, npop):
    '''
    Sample from the N-normal distribution using the harmonisation parameters as th\
e mean values (best case) and the covariance matrix as the N-variance

    # The multivariate normal, multinormal or Gaussian distribution is a
    # generalization of the 1D-normal distribution to higher dimensions.
    # Such a distribution is specified by its mean and covariance matrix.
    # These parameters are analogous to the mean (average or “center”) and
    # variance (standard deviation, or “width,” squared) of the 1D-normal distribu\
tion.

    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.multivaria\
te_normal.html

    # Harmonisation parameters: (nsensor x npar,)
    # Harmonisation parameter uncertainties: (nsensor x npar,)
    # Harmonisation parameter covariance matrix: (nsensor x npar, nsensor x npar)
    # Harmonisation parameter correlation matrix: (nsensor x npar, nsensor x npar)
    # Harmonisation parameter add offsets (internal): (nsensor x npar,)
    # Harmonisation parameter scale factors (internal): (nsensor x npar,)
    # Sensors associated with harmonisation parameters: (nsensor x npar,)
    '''

    a_ave = ds['parameter']
    a_cov = ds['parameter_covariance_matrix']
    draws = np.random.multivariate_normal(a_ave, a_cov, npop)

    return draws

def draws2ensemble(ds, draws, nens, npop):
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

###################################################
# MAIN
###################################################

if __name__ == "__main__":

    #------------------------------------------------------------------------------
    # parser = OptionParser("usage: %prog ch har_file npop nens")
    # (options, args) = parser.parse_args()
    #------------------------------------------------------------------------------
    # ch = args[0]
    # har_file = args[1]
    # npop = args[2]
    # nens_file = args[3]
    #------------------------------------------------------------------------------

    #------------------------------------------------------------------------------
    # RUN PARAMETERS
    #------------------------------------------------------------------------------
    ch = 37
    # /gws/nopw/j04/fiduceo/Users/jmittaz/FCDR/Mike/FCDR_AVHRR/GBCS/dat_cci/
    har_file = 'FIDUCEO_Harmonisation_Data_' + str(ch) + '.nc'    
    npop = 1000000
    nens = 10

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
    if FLAG_mns:
        plotstem = '_'+str(ch)+'_'+'MNS'+'_'+software_tag+'.png'
        mc_file = 'MC_Harmonisation_MNS.nc'
    else:
        plotstem = '_'+str(ch)+'_'+'CRS'+'_'+software_tag+'.png'
        mc_file = 'MC_Harmonisation_CRS.nc'
    #--------------------------------------------------------------------------

    #
    # Load harmonisation parameters
    #

    ds = xarray.open_dataset(har_file, decode_cf=True)
    a_ave = np.array(ds.parameter)
    a_cov = np.array(ds.parameter_covariance_matrix)
    a_u = np.array(ds.parameter_uncertainty) # = np.sqrt(np.diag(a_cov))

    #
    # Load / generate ensemble
    #

    if FLAG_load_mc:

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

        if FLAG_load_draws:
            draws = np_load(filestr_draws)
        else:
            draws = calc_draws(ds, npop)
            np_save(filestr_draws, draws, allow_pickle=False)

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
        da = ensemble - a_ave # [nens,npar]

    else:

        ev = cov2ev(a_cov,c)
        da = ev2da(n,ev) # sum of nPC (99% variance) 

        # da_pc12 = calc_da_pc12(n,ev)
        # da_12 = da_pc12['da_pc1'] + da_pc12['da_pc2'] # sum of first 2 PCs
        # plot_pc_deltas(da_12,a_u)

    da_cov = np.cov(da.T) # should be close to Xcov if working
    da_u = np.sqrt(np.diag(da_cov)) # should be close to Xu if working
    norm_u = np.linalg.norm(a_u - da_u)
    norm_cov = np.linalg.norm(a_cov - da_cov)
    print('|a_u - da_u|=',norm_u)
    print('|a_cov - da_cov|=',norm_cov)

    #
    # Open ensemble file and overwrite channel deltas and uuid:
    #

    if FLAG_write_netcdf:

        ncout = Dataset(mc_file,'r+')
        if ch == 37:
            ncout.variables['delta_params3'][:] = da
            ncout.HARM_UUID3 = ds.uuid
        elif ch == 11:
            ncout.variables['delta_params4'][:] = da
            ncout.HARM_UUID4 = ds.uuid
        else:
            ncout.variables['delta_params5'][:] = da
            ncout.HARM_UUID5 = ds.uuid
        ncout.close()

    # =======================================
    # INCLUDE PLOT CODE:
    exec(open('plot_cov2ensemble.py').read())
    # =======================================

    if FLAG_plot:

        if FLAG_mns:
            plot_ensemble_closure(da,draws,ds) 
            plot_ensemble_decile_selection(Z_norm, ensemble_idx, nens) 
            plot_ensemble_decile_distribution(Z, decile, npop, nens)
        else:
            plot_eigenspectrum(ev)
        plot_ensemble_deltas(da)
        plot_ensemble_deltas_an(da,a_u)
        plot_ensemble_deltas_normalised(da,a_u)
        # plot_crs()

    #------------------------------------------------        
    print('** end')







