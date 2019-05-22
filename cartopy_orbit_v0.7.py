#!/usr/bin/env python

# Version 0.7
# 14 May, 2019
# michael.taylor AT reading DOT ac DOT uk

# PROGRAM: plot_orbit.py

# Copyright (C) 2019 Michael Taylor, University of Reading
# This code was developed for the EC project "Fidelity and Uncertainty in
# Climate Data Records from Earth Observations (FIDUCEO).
# Grant Agreement: 638822
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option)
# any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# A copy of the GNU General Public License should have been supplied along
# with this program; if not, see http://www.gnu.org/licenses/

import os
import os.path
import glob
import sys
from  optparse import OptionParser 
import numpy as np
import numpy.ma as ma
import xarray
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def plot_orbit(filename,projection,figname,titlename):

    thinning  = 2
    ds = xarray.open_dataset(filename)
    ds = ds.isel(ni=slice(None,None,thinning),nj=slice(None,None,thinning))
    x = ds['lon']
    y = ds['lat']
    z = ds['sea_surface_temperature']

    cmap = 'viridis'

    fig  = plt.figure()    
    if projection == 'platecarree':
        p = ccrs.PlateCarree(central_longitude=0)
        threshold = 0
    if projection == 'mollweide':
        p = ccrs.Mollweide(central_longitude=0)
        threshold = 1e6
    if projection == 'robinson':
        p = ccrs.Robinson(central_longitude=0)
        threshold = 0

    ax = plt.axes(projection=p)
    ax.coastlines()
    g = ccrs.Geodetic()
    trans = ax.projection.transform_points(g, x.values, y.values)
    x0 = trans[:,:,0]
    x1 = trans[:,:,1]
    for mask in (x0>threshold,x0<=threshold):
        im = ax.pcolor(ma.masked_where(mask, x0), ma.masked_where(mask, x1), ma.masked_where(mask, z[0,:,:].values), vmin=z.min(),vmax=z.max(), transform=ax.projection, cmap=cmap)
#        im = ax.contourf(ma.masked_where(mask, x0), ma.masked_where(mask, x1), ma.masked_where(mask, z[0,:,:].values), vmin=z.min(),vmax=z.max(), transform=ax.projection, cmap=cmap)

    cb = plt.colorbar(im, orientation="horizontal", extend='both', label='sst [K]')

    if projection == 'platecarree':
        ax.set_extent([-180, 180, -90, 90], crs=p)
        gl = ax.gridlines(crs=p, draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='-')
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlines = True
        gl.ylines = True
        gl.xlocator = mticker.FixedLocator([-180,-120,-60,0,60,120,180])
        gl.ylocator = mticker.FixedLocator([-90,-60,-30,0,30,60,90])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

    plt.title(titlename)
    plt.savefig((figname + ".png"))

if __name__ == "__main__":

    parser = OptionParser("usage: %prog %filename %projection %figname %titlename")
    (options, args) = parser.parse_args()

#   DEFAULTS
#   --------
    filename = '19890527071833-ESACCI-L2P_GHRSST-SSTskin-AVHRR11_G-CDR2.1-v02.0-fv01.0.nc'
    projection = 'platecarree'
#    projection = 'mollweide'
#    projection = 'robinson'
    figstem = "plot_orbit"
    titlename = "L2P: CDR2.1_release"
#    titlename = "L2P: CDR2.1_release: AVHRR11_G 1989-05-27 07:18:33"

    if len(args) == 0:
        path_in = "/Users/michaeltaylor/Desktop/SST_CDR_STATS/temp/plot_orbit/"
#        path_in = "/gws/nopw/j04/fiduceo/Users/mtaylor/sst/"
        nclist = os.path.join(path_in,'*.nc')
        filelist = glob.glob(nclist)
        for i in range(len(filelist)):
            filename = str(filelist[i])
            figname = figstem + ": " + str(i)
            titlename = "L2P: CDR2.1_release" + ": " + str(i)
            plot_orbit(filename,projection,figname,titlename)
    elif len(args) == 1:
        filename = args[0]
        figname = figstem
        plot_orbit(filename,projection,figname,titlename)
    elif len(args) == 2:
        filename = args[0]
        if ( (args[1] == 'platecarree') | (args[1] == 'mollweide') | (args[1] == 'robinson') ):
            filename = args[0]
            projection = args[1]
            figname = figstem
            plot_orbit(filename,projection,figname,titlename)
    elif len(args) == 4:
            filename = args[0]
            projection = args[1]
            figname = args[2]
            titlename = args[3]
            plot_orbit(filename,projection,figname,titlename)

