#!/usr/bin/env python

# call as: python ensemble2bt.py 
# include plot code: plot_cov2ensemble.py
  
# =======================================
# Version 0.1
# 24 July, 2019
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
import crs as crs              # constrained random sampling code: crs.py
import cov2u as cov2u          # estimation of uncertainty from covariance matrix
import convert_func as convert # measurement equations & L<-->BT conversion
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

def calc_dBT(dX, har, mmd, channel, idx_, cci):

    noT = False
    Xave = np.array(har.parameter)
    dBT = np.empty(shape=(len(mmd['avhrr-ma_x']),dX.shape[0]))
    for i in range(dX.shape[0]):
        parameters = dX[i,:] + Xave
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
        WV = 0.0 * Tinst
        if cci:            
            Lict = convert.bt2rad_cci(Tict,channel)
            L = convert.counts2rad_cci(channel,Ce,Cs,Cict,Lict)
            BT = convert.rad2bt_cci(L,channel)[:,3,3]
        else:
            Lict = convert.bt2rad(Tict,channel,lut)
            L = convert.count2rad(Ce,Cs,Cict,Lict,Tinst,WV,channel,a0,a1,a2,a3,a4,noT)
            BT = convert.rad2bt(L,channel,lut)[:,3,3]

        dBT[:,i] = BT

    return dBT

###################################################
# MAIN
###################################################

if __name__ == "__main__":

    #------------------------------------------------------------------------------
    # parser = OptionParser("usage: %prog ch har_file ens_file mmd_file")
    # (options, args) = parser.parse_args()
    #------------------------------------------------------------------------------
    # ch = args[0]
    # har_file = args[1]
    # ens_file = args[2]
    # mmd_file = args[3]
    #------------------------------------------------------------------------------

    #------------------------------------------------------------------------------
    # RUN PARAMETERS
    #------------------------------------------------------------------------------
    ch = 37
    # /gws/nopw/j04/fiduceo/Users/jmittaz/FCDR/Mike/FCDR_AVHRR/GBCS/dat_cci/
    har_file = 'FIDUCEO_Harmonisation_Data_' + str(ch) + '.nc'    
    mmd_file = 'mta_mmd.nc'
    idx = 7 # MTA (see avhrr_sat)
    mns = False
    cci = True
    FLAG_new = True # NEW harmonisation structure (run >= '3.0-4d111a1')
    FLAG_plot = True
#    software_tag = '3e8c463' # job dir=job_avhxx_v6_EIV_10x_11 (old runs)
#    software_tag = '4d111a1' # job_dir=job_avhxx_v6_EIV_10x_11 (new runs)
    software_tag = 'v0.3Bet' # job_dir=job_avhxx_v6_EIV_10x_11 (new runs)
    if mns:
        plotstem = '_'+str(ch)+'_'+'MNS'+'_'+software_tag+'.png'
        ens_file = 'MC_Harmonisation_MNS.nc'
    else:
        plotstem = '_'+str(ch)+'_'+'CRS'+'_'+software_tag+'.png'
        ens_file = 'MC_Harmonisation_CRS.nc'
    #--------------------------------------------------------------------------

    mmd = xarray.open_dataset(mmd_file, decode_times=False)    
    if ch == 37:
        channel = 3
        BT_MMD = mmd['avhrr-ma_ch3b'][:,3,3]    # (55604, 7, 7)
    elif ch == 11:
        channel = 4
        BT_MMD = mmd['avhrr-ma_ch4'][:,3,3]
    else:
        channel = 5
        BT_MMD = mmd['avhrr-ma_ch5'][:,3,3]

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

    #------------------------------------------------

    har = xarray.open_dataset(har_file, decode_cf=True)
    Xave = np.array(har.parameter)
    Xcov = np.array(har.parameter_covariance_matrix)
    Xu = np.array(har.parameter_uncertainty) # = np.sqrt(np.diag(Xcov))

    #
    # Load ensemble
    #

    ncout = Dataset(ens_file,'r')
    if ch == 37:
        dX = ncout.variables['delta_params3'][:]
    elif ch == 11:
        dX = ncout.variables['delta_params4'][:]
    else:
        dX = ncout.variables['delta_params5'][:]
    ncout.close()

    dXcov = np.cov(dX.T) # should be close to Xcov if working
    dXu = np.sqrt(np.diag(dXcov)) # should be close to Xu if working
    norm_u = np.linalg.norm(Xu-dXu)
    norm_cov = np.linalg.norm(Xcov-dXcov)
    print('|Xu - dXu|=',norm_u)
    print('|Xcov - dXcov|=',norm_cov)

    dBT = calc_dBT(dX, har, mmd, channel, idx_, cci)

    # ALTERNATIVE nPC CRITERION:
    #
    # Calc radiance of best-case
    # Calc radiance of Xhat
    # Calc BT of best-case
    # Calc BT of Xhat
    # Cal BT diff
    # if BT diff < 0.001K output n_PC; stop
    # Sample eigenvectors --> 10-member ensemble
    # export ensemble

    # =======================================
    # INCLUDE PLOT CODE:
    exec(open('plot_cov2ensemble.py').read())
    # =======================================

    if FLAG_plot:

        plot_BT_deltas(dBT,BT_MMD)

    #------------------------------------------------        
    print('** end')







