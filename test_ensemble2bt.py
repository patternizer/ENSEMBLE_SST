#!/usr/bin/env python

# call as: python test_ensemble2bt.py 
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
import ensemble_func # functions for ensemble generation
import convert_func as convert # functions for L<-->BT conversion & HAR + CCI Meas Eqn
#------------------------------------------------------------------------------

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
    #--------------------------------------------------------------------------

    if mns:
        plotstem = '_'+str(ch)+'_'+'MNS'+'_'+software_tag+'.png'
        ens_file = 'MC_Harmonisation_MNS.nc'
    else:
        plotstem = '_'+str(ch)+'_'+'CRS'+'_'+software_tag+'.png'
        ens_file = 'MC_Harmonisation_CRS.nc'

    mmd = xarray.open_dataset(mmd_file, decode_times=False)    
    if ch == 37:
        channel = 3
        BT_mmd = mmd['avhrr-ma_ch3b'][:,3,3]    # (55604, 7, 7)
    elif ch == 11:
        channel = 4
        BT_mmd = mmd['avhrr-ma_ch4'][:,3,3]
    else:
        channel = 5
        BT_mmd = mmd['avhrr-ma_ch5'][:,3,3]

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

    lut = convert.read_in_LUT(avhrr_sat[idx])

    #------------------------------------------------

    har = xarray.open_dataset(har_file, decode_cf=True)

    # Load ensemble

    ncout = Dataset(ens_file,'r')
    if ch == 37:
        da = ncout.variables['delta_params3'][:]
    elif ch == 11:
        da = ncout.variables['delta_params4'][:]
    else:
        da = ncout.variables['delta_params5'][:]
    ncout.close()

    # Calculate ensemble BT and best-case BT

    BT_ens = ensemble2BT(da, har, mmd, lut, channel, idx_, cci)
    BT_har = ensemble2BT(da*0., har, mmd, lut, channel, idx_, cci)[:,0]

    # =======================================
    # INCLUDE PLOT CODE:
    exec(open('plot_cov2ensemble.py').read())
    # =======================================

    if FLAG_plot:

        plot_BT_deltas(BT_ens,BT_mmd)

    #------------------------------------------------        
    print('** end')







