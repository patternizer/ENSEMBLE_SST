#!/usr/bin/env python

# call as: python cov2ensemble.py 
# include plot code: plot_cov2ensemble.py
  
# =======================================
# Version 0.12
# 16 July, 2019
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

#------------------------------------------------------------------------------
import crs as crs            # constrained random sampling code: crs.py
import cov2u as cov2u        # estimation of uncertainty from covariance matrix
import convert_func as con   # measurement equations & L<-->BT conversion
#------------------------------------------------------------------------------

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

def calc_dX(n,ev):
    '''
    Create (2*n) deltas of Xave using (un)constrained random sampling
    '''
    nPC = ev['nPC']
    eigenvalues = ev['eigenvalues']
    eigenvectors = ev['eigenvectors']
    nparameters = eigenvectors.shape[1]
    dX_constrained = np.zeros(shape=(2*n,nparameters))
#    random_constrained = np.sort(np.array(crs.generate_10(n)))
    for k in range(nPC):
        random_constrained = np.sort(np.array(crs.generate_10(n)))
        dX_c = np.zeros(shape=(2*n,nparameters))
        for i in range((2*n)):        
            dX_c[i,:] = random_constrained[i] * np.sqrt(eigenvalues[k]) * eigenvectors[:,k]
        dX_constrained = dX_constrained + dX_c
    dX = dX_constrained

    return dX

def calc_dX2(n,ev):
    '''
    Create (2*n) deltas of Xave using (un)constrained random sampling
    '''
    nPC = ev['nPC']
    eigenvalues = ev['eigenvalues']
    eigenvectors = ev['eigenvectors']
    nparameters = eigenvectors.shape[1]
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

    dX2 = {}
    dX2['dX0_constrained'] = np.array(dX0_constrained)
    dX2['dX0_unconstrained'] = np.array(dX0_unconstrained)
    dX2['dX1_constrained'] = np.array(dX1_constrained)
    dX2['dX1_unconstrained'] = np.array(dX1_unconstrained)

    return dX2

def calc_dBT(dA, har, mmd, channel, idx_):

    noT = False
    dBT = np.empty(shape=(len(mmd['avhrr-ma_x']),dA.shape[0]))
    for i in range(dA.shape[0]):
        parameters = dA[i,:]    
        if channel == 3:
            npar = 3
            a0 = parameters[(idx_ *npar)]
            a1 = parameters[(idx_ *npar)+1]
            a2 = parameters[(idx_ *npar)+2]
            a3 = 0.0
            a4 = 0.0
            if noT:
                a2 = 0.0
            Ce = mmd['avhrr-ma_ch3b_earth_counts']
            Cs = mmd['avhrr-ma_ch3b_space_counts']
            Cict = mmd['avhrr-ma_ch3b_bbody_counts']
        elif channel == 4:
            npar = 4
            a0 = parameters[(idx_ *npar)]
            a1 = parameters[(idx_ *npar)+1]
            a2 = parameters[(idx_ *npar)+2]
            a3 = parameters[(idx_ *npar)+3]
            a4 = 0.0
            if noT:
                a3 = 0.0
            Ce = mmd['avhrr-ma_ch4_earth_counts']
            Cs = mmd['avhrr-ma_ch4_space_counts']
            Cict = mmd['avhrr-ma_ch4_bbody_counts']
        else:
            npar = 4
            a0 = parameters[(idx_ *npar)]
            a1 = parameters[(idx_ *npar)+1]
            a2 = parameters[(idx_ *npar)+2]
            a3 = parameters[(idx_ *npar)+3]
            a4 = 0.0
            if noT:
                a3 = 0.0
            Ce = mmd['avhrr-ma_ch5_earth_counts']
            Cs = mmd['avhrr-ma_ch5_space_counts']
            Cict = mmd['avhrr-ma_ch5_bbody_counts']    
        Tict = mmd['avhrr-ma_ict_temp'] # equivalent to mmd['avhrr-ma_orbital_temp']
        T_mean = np.mean(Tict[:,3,3])
        T_sdev = np.std(Tict[:,3,3])
        Tinst = (mmd['avhrr-ma_orbital_temperature'][:,3,3] - T_mean) / T_sdev
        Lict = con.bt2rad(Tict,channel,lut)
        WV = 0.0 * Tinst
        L = con.count2rad(Ce,Cs,Cict,Lict,Tinst,WV,channel,a0,a1,a2,a3,a4,noT)
        BT = con.rad2bt(L,channel,lut)[:,3,3]

        dBT[:,i] = BT

    return dBT

###################################################
# MAIN
###################################################

if __name__ == "__main__":

    #-------------------------------------------------------------------
    # OPTIONS
    #-------------------------------------------------------------------
    ch = 37
    n = 5 # --> (2*n) = 10 = number of ensemble members
    c = 0.99 # variance_threshold
#    N = 10000 # for draw matrix from Xcov
    FLAG_new = True # NEW harmonisation structure (run >= '3.0-4d111a1')
    FLAG_plot = True
    FLAG_dX2 = False
#    software_tag = '3e8c463' # job dir=job_avhxx_v6_EIV_10x_11 (old runs)
#    software_tag = '4d111a1' # job_dir=job_avhxx_v6_EIV_10x_11 (new runs)
    software_tag = 'v0.3Bet' # job_dir=job_avhxx_v6_EIV_10x_11 (new runs)
    plotstem = '_'+str(ch)+'_'+software_tag+'.png'

    # /gws/nopw/j04/fiduceo/Users/jmittaz/FCDR/Mike/FCDR_AVHRR/GBCS/dat_cci/
    har_file = 'FIDUCEO_Harmonisation_Data_' + str(ch) + '.nc'    
    mmd_file = 'mta_mmd.nc'
    ens_file = 'MC_Harmonisation.nc'
    idx = 7 # MTA (see avhrr_sat)
    #-------------------------------------------------------------------

    if ch == 37:
        channel = 3
    elif ch == 11:
        channel = 4
    else:
        channel = 5

    avhrr_sat = [b'N12',b'N14',b'N15',b'N16',b'N17',b'N18',b'N19',b'MTA',b'MTB'] # LUT ordering
    if FLAG_new:
        # RQ: NEW2  AATSR, ATSR2,   MTA,   N19,   N18,   N17,   N16,   N15,   N14,   N12,   N11
        # index         0      1      2      3      4      5      6      7      8      9     10
        # --> new index map for LUT (N12 --> MTA)
        idx_ = 7 - idx + 2
        # RQ: NEW1  AATSR,   MTA,   N19,   N18,   N17,   N16,   N15,   N14,   N12,   N11
        # index         0      1      2      3      4      5      6      7      8      9
        # --> new index map for LUT (N12 --> MTA)
        # idx_ = 7 - idx + 1
    else:
        # RQ: OLD     MTA,   N19,   N18,   N17,   N16,   N15,   N14,   N12,   N11
        # index         0      1      2      3      4      5      6      7      8
        # --> new index map for LUT (N12 --> MTA)
        idx_ = 7 - idx

    lut = con.read_in_LUT(avhrr_sat[idx])

    mmd = xarray.open_dataset(mmd_file, decode_times=False)    
    if channel == 3:
        BT_MMD = mmd['avhrr-ma_ch3b'][:,3,3]    # (55604, 7, 7)
    elif channel == 4:
        BT_MMD = mmd['avhrr-ma_ch4'][:,3,3]
    else:
        BT_MMD = mmd['avhrr-ma_ch5'][:,3,3]

    #------------------------------------------------

    har = xarray.open_dataset(har_file, decode_cf=True)
    Xave = np.array(har.parameter)
    Xcov = np.array(har.parameter_covariance_matrix)
    Xcor = np.array(har.parameter_correlation_matrix)
    Xu = np.array(har.parameter_uncertainty) # = np.sqrt(np.diag(Xcov))

    ev = cov2ev(Xcov,c)
    dX = calc_dX(n,ev) # sum of nPC (99% variance)

    dXcov = np.cov(dX.T) # should be close to Xcov if working
    dXu = dX             # should be close to Xu if working
    print('|Xu - dX|=',np.linalg.norm(Xu-dX))
    print('|Xcov - dXcov|=',np.linalg.norm(Xcov-dXcov))

    #
    # Open ensemble file and overwrite channel deltas and uuid:
    #

    ncout = Dataset(ens_file,'r+')
    if ch == 37:
        ncout.variables['delta_params3'][:] = dX
        ncout.HARM_UUID3 = har.uuid
    elif ch == 11:
        ncout.variables['delta_params4'][:] = dX
        ncout.HARM_UUID4 = har.uuid
    else:
        ncout.variables['delta_params5'][:] = dX
        ncout.HARM_UUID5 = har.uuid
    ncout.close()

    if FLAG_dX2:
        dX2 = calc_dX2(n,ev)
        dX = dX2['dX0_constrained'] + dX2['dX1_constrained'] # sum of first 2 PCs
        plot_pc_deltas(dX,Xu)

    dBT = calc_dBT(dX, har, mmd, channel, idx_)

    # =======================================
    # INCLUDE PLOT CODE:
    exec(open('plot_cov2ensemble.py').read())
    # =======================================

    if FLAG_plot:

        plot_eigenspectrum(ev)
        plot_ensemble_an(dX,Xu)
        plot_ensemble_deltas(dX)
        plot_ensemble_deltas_normalised(dX,Xu)
        plot_BT_deltas(dBT,BT_MMD)
        plot_crs()

    #------------------------------------------------        
    print('** end')







