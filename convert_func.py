import numpy as np
from netCDF4 import Dataset

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def dbyd(a1,a2,b1,b2):
    dbyd = ((a2-a1)/(b2-b1))
    return dbyd

##############################################################
#                                                            #
#  FUNCTIONS FOR CONVERTING BETWEEN BT, COUNTS AND RADIANCE  #
#                                                            #  
#  NB: channel=3 --> 3.7 micron                              #
#  NB: channel=4 -->  11 micron                              #
#  NB: channel=5 -->  12 micron                              #
#                                                            #  
#  read_in_LUT()                                             #
#  REQUIRES avhrr_sat to be in the format:                   #
#  N12,N14,N15,N16,N17,N18,N19,MTA,MTB                       #
#  RETURNS LUT dictionary with groups "L" and "BT"           #
#                                                            #
#  rad2bt()                                                  #
#  REQUIRES array of radiance, channel (either 3, 4 or 5)    #
#  and lut output from read_in_LUT (sensor specific)         #
#  RETURNS BT in array of same initial shape of L            # 
#                                                            # 
#  bt2rad()                                                  #
#  REQUIRES array of BT, channel (either 3, 4 or 5)          #
#  and lut output from read_in_LUT (sensor specific)         #
#  RETURNS L in array of same initial shape of BT            #
#                                                            #
#  dbt_drad()                                                #
#  REQUIRES array of radiance, channel (either 3, 4 or 5)    #
#  and lut output from read_in_LUT (sensor specific)         #
#  RETURNS dbt_drad array in the same shape as L input       #
#                                                            #
#  drad_da()                                                 #
#  REQUIRES arrays of Lict,Ce,Cs,Cict,Tict,Tinst,WV          #
#  Also requires channel and satellite sensor.               #
#  RETURNS: If 3.7um channel, then three derivatives of      #
#  radiance with respect to a2,a4,a5 coefficents. Else       #
#  it returns four derivitives of radiance with respect      #
#  to a2,a3,a4 and a5 coefficents                            #
#                                                            #
#  count2rad()                                               #
#  REQUIRES arrays Ce,Cs,Cict,Lict,Tinst,WV,channel (either  #
#  3, 4 or 5), and harmonisation coeffs a1,a2,a3,a4,a5       #
#  RETURNS radiance in array with same shape as Ce input     #
#                                                            #
#  count2rad_cci()                                           #
#  REQUIRES arrays Ce,Cs,Cict,Lict,channel (3, 4 or 5)       #
#  RETURNS radiance in array with same shape as Ce input     #
#                                                            #
#  rad2bt_cci()                                              #
#  REQUIRES array of radiance, channel (either 3, 4 or 5)    #
#  RETURNS BT in array of same initial shape of L            #
#                                                            #
#  bt2rad_cci()                                              #
#  REQUIRES array of BT, channel (either 3, 4 or 5)          #
#  RETURNS L in array of same initial shape of BT            #
#                                                            #
##############################################################

def read_in_LUT(avhrr_sat):
    LUT = {}
    all_lut_radiance_dict = np.load('lut_radiance.npy', encoding='bytes', allow_pickle=True).item()
    all_lut_BT_dict = np.load('lut_BT.npy', encoding='bytes', allow_pickle=True).item()
    try:
        LUT['L'] = all_lut_radiance_dict[avhrr_sat][:]
        LUT['BT'] = all_lut_BT_dict[avhrr_sat][:]
    except:
        print("Sensor for AVHRR does not exist: ", avhrr_sat)
    return LUT

def rad2bt(L,channel,lut):
    BT = np.interp(L,lut['L'][:,channel],lut['BT'][:,channel],left=-999.9,right=-999.9)
    return BT

def bt2rad(BT,channel,lut):
    L = np.interp(BT,lut['BT'][:,channel],lut['L'][:,channel],left=-999.9,right=-999.9)
    return L

def dbt_drad(L,channel,lut):
    dbtdrad = np.zeros_like(L)
    # Over array L, determine the two values in the LUT either side
    # Then, find dbt by drad and add to new array
    ndim = len(np.shape(L))
    for i in xrange(0,len(L)):
        if ndim > 1.5:
            for j in xrange(0,len(L[0])):
                if ndim > 2.5:
                    for k in xrange(0,len(L[0,0])):
                        element = L[i,j,k]
                        idx = find_nearest(lut['L'][:,channel],element)
                        if lut['L'][idx,channel] > element:
                            dbtdrad[i,j,k] = (dbyd(lut['BT'][idx-1,channel],lut['BT'][idx,channel],lut['L'][idx-1,channel],lut['L'][idx,channel]))
                        else:
                            dbtdrad[i,j,k] = (dbyd(lut['BT'][idx,channel],lut['BT'][idx+1,channel],lut['L'][idx,channel],lut['L'][idx+1,channel]))
                else:
                    element = L[i,j]
                    idx = find_nearest(lut['L'][:,channel],element)
                    if lut['L'][idx,channel] > element:
                        dbtdrad[i,j] = (dbyd(lut['BT'][idx-1,channel],lut['BT'][idx,channel],lut['L'][idx-1,channel],lut['L'][idx,channel]))
                    else:
                        dbtdrad[i,j] = (dbyd(lut['BT'][idx,channel],lut['BT'][idx+1,channel],lut['L'][idx,channel],lut['L'][idx+1,channel]))
        else:
            element = L[i]
            idx = find_nearest(lut['L'][:,channel],element)
            if lut['L'][idx,channel] > element:
                dbtdrad[i] = (dbyd(lut['BT'][idx-1,channel],lut['BT'][idx,channel],lut['L'][idx-1,channel],lut['L'][idx,channel]))
            else:
                dbtdrad[i] = (dbyd(lut['BT'][idx,channel],lut['BT'][idx+1,channel],lut['L'][idx,channel],lut['L'][idx+1,channel]))
    return dbtdrad

def drad_da(Lict,Ce,Cs,Cict,Tict,Tinst,WV,channel,avhrr_sat):
    '''
    Added derivative for WV term. It's a dummy array at present. will need replacing by derivative of f(WV)
    '''
    try:
        if channel == 3:
            drad_da2 = Lict/(Cict - Cs)
            drad_da3 = Tinst
            drad_da4 = WV # dummy array at present. will need replacing by derivative of f(WV)
            return drad_da2,drad_da3,drad_da4
        elif channel > 3:
            drad_da2 = Lict/(Cict - Cs)
            drad_da3 = (Ce - Cict) * (Ce - Cs)
            drad_da4 = Tinst
            drad_da5 = WV # dummy array at present. will need replacing by derivative of f(WV)
            return drad_da2,drad_da3,drad_da4,drad_da5

    except:
        print("No FIDUCEO thermal channel selected: channel=", channel, " < 3")
    
def counts2rad(Ce,Cs,Cict,Lict,Tstar,WV,channel,a0,a1,a2,a3,a4,noT):
    '''
    NB: Tstar = (Tinst - T_mean) / T_std
    NB: WV is a dummy term currently unused
    NB: MTA is a boolean flag: if True --> #105 and #101 (if False --> #106 and #102)
    '''
    L = np.empty(shape=(Ce.shape[0],Ce.shape[1]))
    try:
        if channel == 3:
            if noT:
                L = a0 + ((Lict * (0.985140 + a1)) / (Cict - Cs)) * (Ce - Cs)                             # 105
            L = a0 + ((Lict * (0.985140 + a1)) / (Cict - Cs)) * (Ce - Cs) + a2 * Tstar                    # 106
        elif channel > 3:  
            if noT:
                L = a0 + ((Lict * (0.985140 + a1)) / (Cict - Cs) + a2 * (Ce - Cict)) * (Ce - Cs)          # 101
            L = a0 + ((Lict * (0.985140 + a1)) / (Cict - Cs) + a2 * (Ce - Cict)) * (Ce - Cs) + a3 * Tstar # 102
    except:
        print("No FIDUCEO thermal channel selected: channel=", channel, " < 3")

    return L

def counts2rad_cci(channel,Ce,Cs,Cict,Lict):
    '''
    CCI v1.5 counts --> L measurement equation
    '''
    Nspace = np.append(np.zeros(3),np.array([0,-4.98,-3.4]))
    nonlinear = np.vstack([np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3),np.array([5.44,-0.10152,0.00046964]), np.array([3.84,-0.06249,0.00025239])]).T
    a0 = nonlinear[0,channel]+(1.+nonlinear[1,channel]+nonlinear[2,channel]*Nspace[channel])*Nspace[channel]
    a1 = 1.+nonlinear[1,channel]+2.*nonlinear[2,channel]*Nspace[channel]
    a2 = nonlinear[2,channel]
    s = Cs
    g = (Lict-Nspace[channel])/(Cs-Cict)
    term1 = a0+a1*s*g+a2*s*s*g*g
    term2 = -1.*a1*g-2.*a2*s*g*g
    term3 = a2*g*g
    L = (term1+term2*Ce+term3*Ce*Ce)
    return L

def rad2bt_cci(L,channel):
    '''
    CCI v1.5 L --> BT conversion
    '''
    offset = np.append(np.zeros(3),np.array([-2.0653147,-0.56503332,-0.38472766]))
    slope = np.append(np.zeros(3),np.array([1.0034418,1.0015090,1.0011264]))         
    Central_Wavenumber = np.append(np.zeros(3),np.array([2687.0392,927.2763,837.80762]))
    Planck_C1 = 0.00001191042722
    Planck_C2 = 1.4387752
    coef1 = Planck_C1 * Central_Wavenumber[channel]**3
    coef2 = Planck_C2 * Central_Wavenumber[channel]

    tstar = coef2/np.log(1.+coef1/L)
    BT = offset[channel]+slope[channel]*tstar
    return BT

def bt2rad_cci(BT,channel):
    '''
    CCI v1.5 BT --> L conversion
    '''
    offset = np.append(np.zeros(3),np.array([-2.0653147,-0.56503332,-0.38472766]))
    slope = np.append(np.zeros(3),np.array([1.0034418,1.0015090,1.0011264]))         
    Central_Wavenumber = np.append(np.zeros(3),np.array([2687.0392,927.2763,837.80762]))
    Planck_C1 = 0.00001191042722
    Planck_C2 = 1.4387752
    coef1 = Planck_C1 * Central_Wavenumber[channel]**3
    coef2 = Planck_C2 * Central_Wavenumber[channel]
    tstar = (BT-offset[channel])/slope[channel]
    L = (coef1/(np.exp(coef2/tstar)-1.))
    return L


