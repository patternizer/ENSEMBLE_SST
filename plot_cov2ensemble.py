# !/usr/bin/env python

# Code include segment
# ========================================
# Version 0.5
# 28 July, 2019
# https://patternizer.github.io/
# michael.taylor AT reading DOT ac DOT uk
# ========================================

#---------------------------------------------------------------------------
# PLOT LIST (alphabetical):
#---------------------------------------------------------------------------

# plot_crs(): Constrained randing sampling (CRS) demo
# plot_eigenspectrum(ev): Eigenspectrum + cumulative relative variance with nPC label 
# plot_ensemble_closure(da,draws,har): Harmonisation a_cov and a_u versus ensemble-calculated values
# plot_ensemble_decile_distribution(Z, decile, npop, nens): Draws with decile per parameter
# plot_ensemble_decile_selection(Z_norm, ensemble_idx, nens): Draws with decile selection per parameter
# plot_ensemble_deltas(da): Ensemble deltas / parameter uncertainty
# plot_ensemble_deltas_an(da,a_u): Ensemble deltas / parameter uncertainty per a(n)
# plot_ensemble_deltas_normalised(da,a_u): Ensemble deltas (not normalised)
# plot_ensemble_deltas_pc12(da_pc12, a_u): Project ensemble onto PC1 and PC2
# plot_ensemble_diff_BT_scatterplot(BT_ens,BT_mmd): Ensemble BT versus BT_mmd (in 10K bands)
# plot_ensemble_diff_BT_timeseries(BT_ens,BT): Ensemble BT minus BT [nens,n]
# plot_ensemble_diff_L_timeseries(L_ens,L): Ensemble L minus L [nens,n]
# plot_orbit_var(lat, lon, var, vmin, vmax, projection, filestr, titlestr, varstr): Swathe plot of variable with given lat and lon arrays
 
#---------------------------------------------------------------------------

def plot_crs():
    '''
    Constrained versus unconstrained random sampling demo
    '''
    from ensemble_func import generate_n_single
    from ensemble_func import generate_n

    for n in np.array([10,50,100,500,1000,5000,10000,50000]):

        random_numbers_unconstrained = generate_n_single(n)
        random_numbers_constrained = generate_n(n)

        fig,ax = plt.subplots(1,2)
        labelstr_constrained = 'n=' + str(n) + ': constrained'
        labelstr_unconstrained = 'n=' + str(n) + ': unconstrained'
        ax[0].plot(np.sort(np.array(random_numbers_unconstrained)), label=labelstr_unconstrained)
        ax[0].plot(np.sort(np.array(random_numbers_constrained)), label=labelstr_constrained)
        ax[0].legend(loc=2, fontsize=8)
        ax[0].set_ylim(-5,5)
        ax[0].set_ylabel('z-score')
        ax[0].set_xlabel('rank')
        ax[0].set_title(r'Sorted random sample from $erf^{-1}(x)$')
        ax[1].hist(random_numbers_unconstrained,bins=100,alpha=0.3,label='unconstrained')
        ax[1].hist(random_numbers_constrained,bins=100,alpha=0.3,label='constrained')
        ax[1].set_xlim(-5,5)
        ax[1].set_xlabel('z-score')
        ax[1].set_ylabel('count')
        plotstr = 'random_numbers_n_' + str(n) + plotstem
        plt.tight_layout()
        plt.savefig(plotstr)
        plt.close('all')

def plot_eigenspectrum(ev):
    '''
    Eigenspectrum + cumulative relative variance with nPC label [npar]
    '''
    nPC = ev['nPC']
    fig,ax = plt.subplots()
    plt.plot(ev['eigenvalues_rank'], ev['eigenvalues_norm'], linestyle='-', marker='.', color='b', label=r'$\lambda/sum(lambda)$')
    plt.plot(ev['eigenvalues_rank'][nPC], ev['eigenvalues_norm'][nPC], marker='o', color='k', mfc='none',label=None)
    plt.plot(ev['eigenvalues_rank'], ev['eigenvalues_cumsum'], linestyle='-', marker='.', color='r',label='cumulative')
    labelstr = 'n(PC)='+str(nPC+1)+' var='+"{0:.5f}".format(ev['nPC_variance'])
    plt.plot(ev['eigenvalues_rank'][nPC], ev['eigenvalues_cumsum'][nPC], marker='o', color='k', mfc='none',label=labelstr)
    plt.legend(loc='right', fontsize=10)
    plt.xlabel('rank')
    plt.ylabel('relative variance')
    plotstr = 'eigenspectrum' + plotstem
    plt.tight_layout()
    plt.savefig(plotstr)
    plt.close('all')

def plot_ensemble_closure(da,draws,har):
    '''
    Harmonisation a_cov and a_u versus ensemble-calculated values [npar,npar] & [npar]
    '''
    a_u = np.array(har.parameter_uncertainty)
    a_cov = np.array(har.parameter_covariance_matrix)
    da_cov = np.cov(draws.T) # [npar,npar] 
    da_u = np.sqrt(np.diag(da_cov)) # [npar]
    norm_u = np.linalg.norm(a_u - da_u)
    norm_cov = np.linalg.norm(a_cov - da_cov)

    umin = np.min([a_u,da_u])
    umax = np.max([a_u,da_u])
    covmin = np.min([a_cov,da_cov])
    covmax = np.max([a_cov,da_cov])

    fig,ax = plt.subplots(2,2)
    g = sns.heatmap(a_cov - da_cov,ax=ax[0,0])
    ax[0,0].set_xlabel('parameter, a(n)')
    ax[0,0].set_ylabel('parameter, a(n)')
    ax[0,1].plot(np.arange(1,len(a_u)+1), a_u - da_u,'k.',markersize=10,alpha=0.2)
    ax[0,1].set_xlabel('parameter, a(n)')
    ax[0,1].set_ylabel('HAR-MNS: u(n)')
    ax[1,0].plot(a_cov.ravel(), da_cov.ravel(),'k.',markersize=10,alpha=0.2)
    ax[1,0].plot([covmin,covmax],[covmin,covmax], '-', color='red')
    ax[1,0].set_xlabel('HAR: cov(n,n)')
    ax[1,0].set_ylabel('MNS: cov(n,n)')
    ax[1,1].plot(a_u, da_u,'k.',markersize=10,alpha=0.2)
    ax[1,1].plot([umin,umax],[umin,umax], '-', color='red')
    ax[1,1].set_xlabel('HAR: u(n)')
    ax[1,1].set_ylabel('MNS: u(n)')
    plotstr = 'ensemble_closure' + plotstem
    plt.tight_layout()
    plt.savefig(plotstr)
    plt.close('all')

def plot_ensemble_decile_distribution(Z, decile, npop, nens):
    '''
    Draws with decile per parameter
    '''
    npar = decile.shape[1]
    if npop > 10000:
        krange = np.linspace(0,npop-1,10000).astype('int')
    else:
        krange = range(npop)
    fig,ax = plt.subplots()
    for k in krange:
        plt.plot(np.arange(1,npar+1),Z[k,:],'.',alpha=0.2)
    for i in range(nens):
        plt.plot(np.arange(1,npar+1),decile[i,:],'-',alpha=1.0,label='decile('+\
str(i+1)+')')
    plt.ylim(-5,5)
    plt.xlabel('harmonisation parameter')
    plt.ylabel('multinormal draw z-score')
    plt.legend(loc=2,fontsize=8, ncol=5)
    plotstr = 'ensemble_decile_distribution' + plotstem
    plt.tight_layout()
    plt.savefig(plotstr)
    plt.close('all')

def plot_ensemble_decile_selection(Z_norm, ensemble_idx, nens):
    '''
    Draws with decile selection per parameter
    '''
    fig,ax = plt.subplots()
    for k in range(nens):
        labelstr = 'decile('+str(k+1)+')'
        plt.plot(Z_norm[k,:],label=labelstr)
        plt.plot(ensemble_idx[k],Z_norm[k,ensemble_idx[k]],marker='o',color='k'\
,mfc='none',label=None)
    plt.ylim(0,25)
    plt.xlabel('multinormal draw')
    plt.ylabel(r'norm distance of multinormal draw from $k^{th}$ decile')
    plt.legend(loc=2,fontsize=8, ncol=5)
    plotstr = 'ensemble_decile_selection.png' + plotstem
    plt.tight_layout()
    plt.savefig(plotstr)
    plt.close('all')

def plot_ensemble_deltas(da):
    '''
    Ensemble deltas (not normalised) [nens,npar]
    '''
    n = int(da.shape[0]/2)
    npar = da.shape[1]
    fig,ax = plt.subplots()
    for i in range(2*n):
        labelstr_c = 'ens(' + str(i+1) + ')'
        plt.plot(np.arange(1,npar+1), da[i,:], lw=2, label=labelstr_c)
    if n <= 5:
        plt.legend(loc=2, fontsize=8, ncol=5)
    plt.ylim(-0.0020,0.0020)
    plt.xlabel('parameter, a(n)')
    plt.ylabel(r'$\delta a(n)$')
    plotstr = 'npc_deltas' + plotstem
    plt.tight_layout()
    plt.savefig(plotstr)
    plt.close('all')
    
def plot_ensemble_deltas_an(da, a_u):
    '''
    Ensemble deltas / parameter uncertainty by parameter a(n) [nens,nsensor] 
    '''
    n = int(da.shape[0]/2)
    nensemble = da.shape[0]
    nparameters = da.shape[1]
    if nparameters > 27:
        for i in range(4):
            fig,ax = plt.subplots()
            idx = np.arange(i,nparameters-1,4) # -1 --> MTA:N12 (excl N11)
            for k in range(len(idx)-1):
                for l in range(nensemble):
                    labelstr = 'ens('+str(l+1)+')'
                    if k == 0:
                        plt.plot(k, da[l,idx[k]] / a_u[idx[k]],'.',label=labelstr)
                    else:
                        plt.plot(k, da[l,idx[k]] / a_u[idx[k]],'.',label=None)
            if n <= 5:
                plt.legend(loc=2, fontsize=8, ncol=5)
            plt.ylim(-5,5)
            plt.ylabel(r'$\delta a(n)/u(n)$')
            plt.xlabel('sensor')
            plotstr = 'ensemble_a' + str(i) + plotstem
            plt.tight_layout()
            plt.savefig(plotstr)
            plt.close('all')
    else:
        for i in range(3):
            fig,ax = plt.subplots()
            idx = np.arange(i,nparameters-1,3) # -1 --> MTA:N12 (excl N11)
            for k in range(len(idx)-1):
                for l in range(nensemble):
                    labelstr = 'ens('+str(l+1)+')'
                    if k == 0:
                        plt.plot(k, da[l, idx[k]] / a_u[idx[k]],'.',label=labelstr)
                    else:
                        plt.plot(k, da[l, idx[k]] / a_u[idx[k]],'.',label=None)
            if n <= 5:
                plt.legend(loc=2, fontsize=8, ncol=5)
            plt.ylim(-5,5)
            plt.ylabel(r'$\delta a(n)/u(n)$')
            plt.xlabel('sensor')
            plotstr = 'ensemble_a' + str(i) + plotstem
            plt.tight_layout()
            plt.savefig(plotstr)
            plt.close('all')

def plot_ensemble_deltas_normalised(da, a_u):
    '''
    Ensemble deltas / parameter uncertainty [nens,npar] 
    '''
    n = int(da.shape[0]/2)
    npar = da.shape[1]
    fig,ax = plt.subplots()
    for i in range(2*n):
        labelstr_c = 'ens(' + str(i+1) + ')'
        plt.plot(np.arange(1,npar+1), da[i,:] / a_u, lw=2, label=labelstr_c)
    if n <= 5:
        plt.legend(loc=2, fontsize=8, ncol=5)
    plt.ylim(-5,5)
    plt.xlabel('parameter, a(n)')
    plt.ylabel(r'$\delta a(n)/u(n)$')
    plotstr = 'npc_deltas_over_Xu' + plotstem
    plt.tight_layout()
    plt.savefig(plotstr)
    plt.close('all')

def plot_ensemble_deltas_pc12(da_pc12, a_u):
    '''
    Project ensemble onto PC1 and PC2
    '''
    n = int(da_pc12['da_pc1'].shape[0]/2)
    fig,ax = plt.subplots()
    for i in range(2*n):
        labelstr = 'PC1: ens(' + str(i+1) + ')'
        plt.plot(da_pc12['da_pc1'][i,:] / a_u, lw=2, label=labelstr)
    if n <= 5:
        plt.legend(loc=2, fontsize=6, ncol=2)
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.xlabel('parameter, a(n)')
    plt.ylabel(r'$\delta a(n)/u(n)$')
    plotstr = 'pc1_deltas_over_Xu' + plotstem
    plt.tight_layout()
    plt.savefig(plotstr)
    plt.close('all')

    fig,ax = plt.subplots()
    for i in range(2*n):
        labelstr = 'PC1: ens(' + str(i+1) + ')'
        plt.plot(da_pc12['da_pc1'][i,:] / a_u, lw=2, label=labelstr)
    if n <= 5:
        plt.legend(loc=2, fontsize=6, ncol=2)
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.xlabel('parameter, a(n)')
    plt.ylabel(r'$\delta a(n)/u(n)$')
    plotstr = 'pc2_deltas_over_Xu' + plotstem
    plt.tight_layout()
    plt.savefig(plotstr)
    plt.close('all')

def plot_ensemble_diff_BT_scatterplot(BT_ens,BT_mmd):
    '''
    Ensemble BT versus BT_mmd (in 10K bands) [nens,n_mmd]
    '''
    n = int(BT_ens.shape[1]/2)
    fig,ax = plt.subplots()
    for i in range(2*n):
        labelstr = 'ens(' + str(i+1) + ')'
        gd = BT_ens[:,i] > 0
        plt.plot(BT_mmd[gd], BT_ens[gd,i], '.', markersize=2, alpha=0.2, label=labelstr)
    plt.plot([220,310],[220,310], '--', color='black', label=None)
    plt.xlim(220,310)
    plt.ylim(220,310)
    if n <= 5:
        plt.legend(loc=2, fontsize=8, ncol=5)
    plt.xlabel(r'brightness temperature, BT / $K$')
    plt.ylabel(r'ensemble brightness temperature, ens(BT) / $K$')
    plotstr = 'bt_deltas' + plotstem
    plt.tight_layout()
    plt.savefig(plotstr)
    plt.close('all')

    BT_vec = np.arange(230,310,10)
    for k in range(len(BT_vec)-1):
        fig,ax = plt.subplots()
        for i in range(2*n):
            labelstr = 'ens(' + str(i+1) + ')'
            domain = (BT_mmd >= BT_vec[k]) & (BT_mmd < BT_vec[k+1])  
            gd = (BT_ens[:,i] > 0) & domain 
            plt.plot(BT_mmd[gd],BT_ens[gd,i], '.', markersize=2, alpha=0.2, label=labelstr)
        plt.plot([BT_vec[k],BT_vec[k+1]],[BT_vec[k],BT_vec[k+1]], '--', color='black', label=None)
        plt.xlim(BT_vec[k],BT_vec[k+1])
        plt.ylim(BT_vec[k],BT_vec[k+1])
        if n <= 5:
            plt.legend(loc=2, fontsize=8, ncol=5)
        plt.xlabel(r'brightness temperature, BT / $K$')
        plt.ylabel(r'ensemble brightness temperature, ens(BT) / $K$')
        plotstr = 'bt_deltas' + '_' + str(BT_vec[k]) + '_' + str(BT_vec[k+1]) + plotstem
        plt.tight_layout()
        plt.savefig(plotstr)
        plt.close('all')

def plot_ensemble_diff_BT_timeseries(BT_ens,BT):
    '''
    Ensemble BT minus BT [nens,n]
    '''
    n = int(BT_ens.shape[1]/2)
    fig, ax = plt.subplots()
    for k in range(2*n):
        label_str = 'Ens(' + str(k+1) + ')'
        plt.plot(BT_ens[:,k] - BT, linewidth=1.0, label=label_str)
    plt.legend(fontsize=10, ncol=1)
    ax.set_ylabel('BT difference / K', fontsize=12)
    plotstr = 'bt_deltas' + plotstem
    plt.tight_layout()
    plt.savefig(plotstr)
    plt.close('all')

def plot_ensemble_diff_L_timeseries(L_ens,L):
    '''
    Ensemble L versus L [nens,n]
    '''
    n = int(L_ens.shape[1]/2)
    fig, ax = plt.subplots()
    for k in range(2*n):
        label_str = 'Ens(' + str(k+1) + ')'
        plt.plot(L_ens[:,k] - L, linewidth=1.0, label=label_str)
    plt.legend(fontsize=10, ncol=1)
    ax.set_ylabel('Radiance difference', fontsize=12)
    plotstr = 'l_deltas' + plotstem
    plt.tight_layout()
    plt.savefig(plotstr)
    plt.close('all')

def plot_orbit_var(lat, lon, var, vmin, vmax, projection, filestr, titlestr, varstr):
    '''
    Swathe plot of variable with given lat and lon arrays
    '''
    x = lon[::10,::10]
    y = lat[::10,::10]
    z = var[::10,::10]

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
    # trans = ax.projection.transform_points(g, x.values, y.values)
    trans = ax.projection.transform_points(g, x, y)
    x0 = trans[:,:,0]
    x1 = trans[:,:,1]

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

#        im = ax.pcolor(x, y, z, transform=ax.projection, cmap=cmap)
        for mask in (x0>threshold,x0<=threshold):

            im = ax.pcolor(ma.masked_where(mask, x), ma.masked_where(mask, y), ma.masked_where(mask, z), vmin=vmin, vmax=vmax, transform=ax.projection, cmap='seismic')

    else:

        for mask in (x0>threshold,x0<=threshold):

            im = ax.pcolor(ma.masked_where(mask, x0), ma.masked_where(mask, x1), ma.masked_where(mask, z), vmin=vmin, vmax=vmax, transform=ax.projection, cmap='seismic')

    cb = plt.colorbar(im, orientation="horizontal", extend='both', label=varstr)

    plt.title(titlestr)
    plt.savefig(filestr)
    plt.close('all')

