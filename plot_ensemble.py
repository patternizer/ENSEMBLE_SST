#!/usr/bin/env python

# ipdb> import os; os._exit(1)

# Code include segment in: calc_ensemble.py
# =======================================
# Version 0.15
# 22 May, 2019
# michael.taylor AT reading DOT ac DOT uk
# =======================================

def plot_bestcase_parameters(ds, sensor):
    '''
    Plot harmonisation parameters and uncertainties for best-case
    '''

    parameter = ds['parameter'] 
    parameter_uncertainty = ds['parameter_uncertainty'] 
    npar = int( len(parameter) / len(sensor) )

    Y = np.array(parameter)
    U = np.array(parameter_uncertainty)

    if npar == 3:

#        Z = a.reshape((len(Y), npar))

        idx0 = np.arange(0, len(Y), 3)        
        idx1 = np.arange(1, len(Y), 3)        
        idx2 = np.arange(2, len(Y), 3) 
        Y0 = []
        Y1 = []
        Y2 = []
        U0 = []
        U1 = []
        U2 = []
        for i in range(0,len(sensor)): 

            k0 = idx0[i]
            k1 = idx1[i]
            k2 = idx2[i]
            Y0.append(Y[k0])
            Y1.append(Y[k1])
            Y2.append(Y[k2])
            U0.append(U[k0])
            U1.append(U[k1])
            U2.append(U[k2])
        Y = np.array([Y0,Y1,Y2])    
        U = np.array([U0,U1,U2])    
        dY = pd.DataFrame({'a(0)': Y0, 'a(1)': Y1, 'a(2)': Y2}, index=list(sensor))                  
        dU = pd.DataFrame({'a(0)': U0, 'a(1)': U1, 'a(2)': U2}, index=list(sensor)) 

        ax = dY.plot(kind="bar", yerr=dU, colormap='viridis', subplots=True, layout=(3, 1), sharey=False, sharex=True, rot=90, fontsize=12, legend=False)

    elif npar == 4:

        idx0 = np.arange(0, len(Y), 4)        
        idx1 = np.arange(1, len(Y), 4)        
        idx2 = np.arange(2, len(Y), 4) 
        idx3 = np.arange(3, len(Y), 4) 
        df = pd.DataFrame()
        Y0 = []
        Y1 = []
        Y2 = []
        Y3 = []
        U0 = []
        U1 = []
        U2 = []
        U3 = []
        for i in range(0,len(sensor)): 

            k0 = idx0[i]
            k1 = idx1[i]
            k2 = idx2[i]
            k3 = idx3[i]
            Y0.append(Y[k0])
            Y1.append(Y[k1])
            Y2.append(Y[k2])
            Y3.append(Y[k3])
            U0.append(U[k0])
            U1.append(U[k1])
            U2.append(U[k2])
            U3.append(U[k3])
        Y = np.array([Y0,Y1,Y2,Y3])    
        U = np.array([U0,U1,U2,U3])    
        dY = pd.DataFrame({'a(0)': Y0, 'a(1)': Y1, 'a(2)': Y2, 'a(3)': Y3}, index=list(sensor))                  
        dU = pd.DataFrame({'a(0)': U0, 'a(1)': U1, 'a(2)': U2, 'a(3)': U3}, index=list(sensor))                  

        ax = dY.plot(kind="bar", yerr=dU, colormap='viridis', subplots=True, layout=(4, 1), sharey=False, sharex=True, rot=90, fontsize=12, legend=False)
    
    plt.tight_layout()
    file_str = "bestcase_parameters.png"
    plt.savefig(file_str)    
    plt.close()

def plot_bestcase_covariance(ds):
    '''
    Plot harmonisation parameter covariance matrix for best-case as a heatmap
    '''

    parameter_covariance_matrix = ds['parameter_covariance_matrix'] 

    Y = np.array(parameter_covariance_matrix)

    fig = plt.figure()
    sns.heatmap(Y, center=0, linewidths=.5, cmap="viridis", cbar=True, vmin=-1.0e-9, vmax=1.0e-6, cbar_kws={"extend":'both', "format":ticker.FuncFormatter(fmt)})
    title_str = "Covariance matrix (relative): max=" + "{0:.3e}".format(Y.max())
    plt.title(title_str)
    plt.savefig('bestcase_covariance_matrix.png')    
    plt.close()

def plot_population_histograms(ds, draws, sensor, nens):
    '''
    Plot histograms of population coefficient z-scores
    '''

    parameter = ds['parameter'] 
    parameter_covariance_matrix = ds['parameter_covariance_matrix'] 
    npar = int( len(parameter) / len(sensor) )

    draws_ave = draws.mean(axis=0)
    draws_std = draws.std(axis=0)
    Z = (draws - draws_ave) / draws_std
#    nbins = nens
    nbins = 100

    #
    # Histograms of ensemble variability
    #

    for i in range(0,len(sensor)):

        fig, ax = plt.subplots(npar,1,sharex=True)
        for j in range(0,npar):

            k = (npar*i)+j
            hist, bins = np.histogram(Z[:,k], bins=nbins, density=False) 
            hist = hist/hist.sum()
            ax[j].fill_between(bins[0:-1], hist, step="post", alpha=0.4)
            ax[j].plot(bins[0:-1], hist, drawstyle='steps-post')
            ax[j].plot((0,0), (0,hist.max()), 'r-')   
            ax[j].tick_params(labelsize=10)
            ax[j].set_xlim([-6,6])
            ax[j].set_ylabel('Prob. density', fontsize=10)
            title_str = sensor[i] + ": a(" + str(j) + ")=" + "{0:.3e}".format(draws_ave[k])
            ax[j].set_title(title_str, fontsize=10)
            
        plt.xlabel(r'z-score', fontsize=10)
        file_str = "population_histograms_" + sensor[i] + ".png"
        plt.savefig(file_str)    

    plt.close('all')

def plot_population_coefficients(ds, draws, sensor, npop):
    '''
    Plot population coefficient z-scores
    '''

    parameter = ds['parameter'] 
    parameter_covariance_matrix = ds['parameter_covariance_matrix'] 
    npar = int( len(parameter) / len(sensor) )

    draws_ave = draws.mean(axis=0)
    draws_std = draws.std(axis=0)
    Z = (draws - draws_ave) / draws_std

    #
    # Ensemble coefficients plotted as z-scores relative to best value
    #

    for i in range(0,len(sensor)):

        fig, ax = plt.subplots(npar,1,sharex=True)
        for j in range(0,npar):

            k = (npar*i)+j
            ax[j].plot(np.arange(0,npop), Z[:,k])            
            ax[j].plot((0,npop), (0,0), 'r-')   
            ax[j].tick_params(labelsize=10)
            ax[j].set_ylabel(r'z-score', fontsize=10)
            ax[j].set_ylim([-6,6])
            title_str = sensor[i] + ": a(" + str(j) + ")=" + "{0:.3e}".format(draws_ave[k])
            ax[j].set_title(title_str, fontsize=10)

        plt.xlabel('Population member', fontsize=10)
        file_str = "population_coefficients_" + sensor[i] + ".png"
        plt.savefig(file_str)    

    plt.close('all')

def plot_population_cdf(ds, draws, sensor, nens, npop):
    '''
    Extract decile values of population
    '''

    parameter = ds['parameter'] 
    npar = int( len(parameter) / len(sensor) )

    draws_ave = draws.mean(axis=0)
    draws_std = draws.std(axis=0)
    Z = (draws - draws_ave) / draws_std

    F = np.array(range(0,npop))/float(npop)    
    Z_est = np.empty(shape=(nens,len(parameter)))

    for i in range(0,npar):

        fig, ax = plt.subplots()
        for j in range(0,len(sensor)):

            k = (npar*i)+j

            #
            # Empirical CDF
            #

            hist, edges = np.histogram( Z[:,k], bins = nens, density = True )
            binwidth = edges[1] - edges[0]
            Z_cdf = np.sort(Z[:,k])
            Z_est[:,k] = np.cumsum(hist) * binwidth
            F_est = edges[1:]
            label_str = sensor[j]
#            plt.plot(F_est, Z_est[:,k], marker='.', linewidth=0.25, label=label_str)
            plt.plot(F_est, Z_est[:,k], linewidth=0.25, label=label_str)
            plt.xlim([-6,6])
            plt.ylim([0,1])
            plt.xlabel('z-score')
            plt.ylabel('Cumulative distribution function (CDF)')
            title_str = 'Harmonisation coefficient: a(' + str(i) + ')' 
            plt.title(title_str)
            plt.legend(fontsize=10, ncol=1)
            file_str = "population_cdf_coefficient_" + str(i) + ".png"
            plt.savefig(file_str)    

    plt.close('all')

def plot_radiance_deltas(L, L_delta, nens, nch):

    fig, ax = plt.subplots()
    for k in range(nens):

        label_str = 'Ens(' + str(k+1) + ')'
        plt.plot(L - L_delta[:,k], linewidth=1.0, label=label_str)

    plt.legend(fontsize=10, ncol=1)
    ax.set_ylabel('Radiance difference', fontsize=12)
    plt.tight_layout()
    file_str = "radiance_ensemble_" + str(nch) + ".png"
    plt.savefig(file_str)
    plt.close('all')

def plot_bt_deltas(BT, BT_delta, nens, nch):

    fig, ax = plt.subplots()
    for k in range(nens):

        label_str = 'Ens(' + str(k+1) + ')'
        plt.plot(BT - BT_delta[:,k], linewidth=1.0, label=label_str)

    plt.legend(fontsize=10, ncol=1)
    ax.set_ylabel('BT difference', fontsize=12)
    plt.tight_layout()
    file_str = "bt_ensemble_" + str(nch) + ".png"
    plt.savefig(file_str)
    plt.close('all')

def plot_ensemble_deltas(ds, ensemble, sensor, nens):
    '''
    Plot ensemble member coefficients normalised to parameter uncertainty
    '''

    parameter = ds['parameter'] 
    parameter_uncertainty = ds['parameter_uncertainty'] 
    npar = int(len(parameter) / len(sensor))

    Y = np.array(parameter)
    U = np.array(parameter_uncertainty)
    Z = np.array(ensemble)

    if npar == 3:

        idx0 = np.arange(0, len(Y), 3)        
        idx1 = np.arange(1, len(Y), 3)        
        idx2 = np.arange(2, len(Y), 3) 
        Y0 = []
        Y1 = []
        Y2 = []
        U0 = []
        U1 = []
        U2 = []
        Z0 = []
        Z1 = []
        Z2 = []
        for i in range(0,len(sensor)): 

            k0 = idx0[i]
            k1 = idx1[i]
            k2 = idx2[i]
            Y0.append(Y[k0])
            Y1.append(Y[k1])
            Y2.append(Y[k2])
            U0.append(U[k0])
            U1.append(U[k1])
            U2.append(U[k2])
            Z0.append(Z[:,k0])
            Z1.append(Z[:,k1])
            Z2.append(Z[:,k2])
        Y = np.array([Y0,Y1,Y2])    
        U = np.array([U0,U1,U2])    
        Z = np.array([Z0,Z1,Z2])    
        dY = pd.DataFrame({'a(0)': Y0, 'a(1)': Y1, 'a(2)': Y2}, index=list(sensor))        
        dU = pd.DataFrame({'a(0)': U0, 'a(1)': U1, 'a(2)': U2}, index=list(sensor))             
        dZ = pd.DataFrame({'a(0)': Z0, 'a(1)': Z1, 'a(2)': Z2}, index=list(sensor))                  

    elif npar == 4:

        idx0 = np.arange(0, len(Y), 4)        
        idx1 = np.arange(1, len(Y), 4)        
        idx2 = np.arange(2, len(Y), 4) 
        idx3 = np.arange(3, len(Y), 4) 
        Y0 = []
        Y1 = []
        Y2 = []
        Y3 = []
        U0 = []
        U1 = []
        U2 = []
        U3 = []
        Z0 = []
        Z1 = []
        Z2 = []
        Z3 = []
        for i in range(0,len(sensor)): 

            k0 = idx0[i]
            k1 = idx1[i]
            k2 = idx2[i]
            k3 = idx3[i]
            Y0.append(Y[k0])
            Y1.append(Y[k1])
            Y2.append(Y[k2])
            Y3.append(Y[k3])
            U0.append(U[k0])
            U1.append(U[k1])
            U2.append(U[k2])
            U3.append(U[k3])
            Z0.append(Z[:,k0])
            Z1.append(Z[:,k1])
            Z2.append(Z[:,k2])
            Z3.append(Z[:,k3])
        Y = np.array([Y0,Y1,Y2,Y3])    
        U = np.array([U0,U1,U2,U3])    
        Z = np.array([Z0,Z1,Z2,Z3])    
        dY = pd.DataFrame({'a(0)': Y0, 'a(1)': Y1, 'a(2)': Y2, 'a(3)': Y3}, index=list(sensor))
        dU = pd.DataFrame({'a(0)': U0, 'a(1)': U1, 'a(2)': U2, 'a(3)': U3}, index=list(sensor)) 
        dZ = pd.DataFrame({'a(0)': Z0, 'a(1)': Z1, 'a(2)': Z2, 'a(3)': Z3}, index=list(sensor))              

    #
    # Lineplot per sensor of ensemble for each parameter
    #

    xs = np.arange(nens)
    for i in range(0,npar):

        fig, ax = plt.subplots()
        for j in range(0,len(sensor)):
                  
            if i == 0:
                ys = (dZ['a(0)'][j] - dY['a(0)'][j]) / dU['a(0)'][j]
            elif i == 1:
                ys = (dZ['a(1)'][j] - dY['a(1)'][j]) / dU['a(1)'][j] 
            elif i == 2:
                ys = (dZ['a(2)'][j] - dY['a(2)'][j]) / dU['a(2)'][j] 
            elif i == 3:
                ys = (dZ['a(3)'][j] - dY['a(3)'][j]) / dU['a(3)'][j] 
            plt.plot(xs, np.sort(ys), marker='.', linewidth=0.5, label=sensor[j])

        ax.set_xlabel('Ensemble member', fontsize=12)
        ax.set_ylabel('Delta / Uncertainty', fontsize=12)
        ax.set_ylim([-6,6])
        title_str = 'Harmonisation coefficient: a(' + str(i) + ')'
        ax.set_title(title_str, fontsize=12)
        ax.legend(fontsize=8)
        file_str = "ensemble_delta_uncertainty_coefficient_" + str(i) + ".png"
        plt.savefig(file_str)    

    plt.close('all')

    #
    # Boxplot per sensor of ensemble for each parameter
    #

    xs = np.arange(nens)
    for i in range(0,npar):

        fig = plt.figure()
        ax = plt.subplot(111)
        ys_all = []
        for j in range(0,len(sensor)):
                  
            if i == 0:
                ys = (dZ['a(0)'][j] - dY['a(0)'][j]) / dU['a(0)'][j]
            elif i == 1:
                ys = (dZ['a(1)'][j] - dY['a(1)'][j]) / dU['a(1)'][j] 
            elif i == 2:
                ys = (dZ['a(2)'][j] - dY['a(2)'][j]) / dU['a(2)'][j] 
            elif i == 3:
                ys = (dZ['a(3)'][j] - dY['a(3)'][j]) / dU['a(3)'][j] 
            ys_all.append(ys)

        ax.boxplot(ys_all, notch=False, sym="o", labels=list(sensor))
        ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
        ax.set_ylabel('Delta / Uncertainty (median statistics)', fontsize=12)
        ax.set_ylim([-6,6])
        title_str = 'Harmonisation coefficient: a(' + str(i) + ')'
        ax.set_title(title_str, fontsize=12)
        file_str = "ensemble_boxplot_delta_uncertainty_coefficient_" + str(i) + ".png"
#        plt.gcf().subplots_adjust(bottom=0.15)
        plt.tight_layout()
        plt.savefig(file_str)    

    plt.close('all')

    #
    # Correlation between ensemble member coefficients
    #

    for i in range(0,npar):

        r = np.empty(shape=(len(sensor),len(sensor)))

        for j in range(0,len(sensor)):

            for k in range(0,len(sensor)):
                if i == 0:
                    yj = dZ['a(0)'][j]
                    yk = dZ['a(0)'][k]
                elif i == 1:
                    yj = dZ['a(1)'][j]
                    yk = dZ['a(1)'][k]
                elif i == 2:
                    yj = dZ['a(2)'][j]
                    yk = dZ['a(2)'][k]
                elif i == 3:
                    yj = dZ['a(3)'][j]
                    yk = dZ['a(3)'][k]

                r[j,k] = np.corrcoef(np.sort(yj), np.sort(yk))[0,1]
                r[k,j] = np.corrcoef(np.sort(yj), np.sort(yk))[1,0]

        fig, ax = plt.subplots()

        #
        # Mask out upper triangle
        #

        mask = np.zeros_like(r)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):

            g = sns.heatmap(r, mask=mask, square=False, annot=True, linewidths=0.5, cmap="viridis", cbar=True, cbar_kws={'label': 'Correlation Coeff.', 'orientation': 'vertical'}, xticklabels=sensor, yticklabels=sensor)
            g.set_yticklabels(g.get_yticklabels(), rotation =0)
            g.set_xticklabels(g.get_yticklabels(), rotation =90)
            title_str = 'Harmonisation coefficient: a(' + str(i) + ')'
            ax.set_title(title_str, fontsize=12)
            plt.tight_layout()
            file_str = "ensemble_correlation_coefficient_" + str(i) + ".png"
            plt.savefig(file_str)    

    plt.close('all')

def plot_eigenval(eigenval, title_str, file_str):
    '''
    Plot eigenvalues as a scree plot
    '''

    Y = eigenval / max(eigenval)

    fig = plt.figure()
    plt.fill_between( np.arange(0,len(Y)), Y, step="post", alpha=0.4)
    plt.plot( np.arange(0,len(Y)), Y, drawstyle='steps-post')
    plt.tick_params(labelsize=12)
    plt.ylabel("Relative value", fontsize=12)
    plt.title(title_str)
    plt.savefig(file_str)
    plt.close()

def plot_eigenvec(eigenvec, title_str, file_str):
    '''
    Plot eigenvector matrix as a heatmap
    '''

    X = eigenvec

    fig = plt.figure()
    sns.heatmap(X, center=0, linewidths=.5, cmap="viridis", cbar=True)
    plt.title(title_str)
    plt.savefig(file_str)
    plt.close()

def plot_ensemble_check(ds, ensemble):
    '''
    Calculate correlation matrix of ensemble.
    Calculate covariance matrix of ensemble.
    Calculate diff from harmonisation.
    Eigenvalue scree plot of covariance and correlation matrices.
    '''

    cov_par = ds.parameter_covariance_matrix
    cov_ensemble = np.cov(ensemble, rowvar=False)
    cov_diff = cov_par - cov_ensemble

    corr_par = ds.parameter_correlation_matrix
    corr_ensemble = np.corrcoef(ensemble, rowvar=False)
    corr_diff = corr_par - corr_ensemble

    cov_par_eigenval, cov_par_eigenvec = calc_eigen(cov_par)
    cov_ensemble_eigenval, cov_ensemble_eigenvec = calc_eigen(cov_ensemble)

    corr_par_eigenval, corr_par_eigenvec = calc_eigen(corr_par)
    corr_ensemble_eigenval, corr_ensemble_eigenvec = calc_eigen(corr_ensemble)

    #
    # Plot Eigenvalues
    #

    title_str = 'Scree plot (correlation matrix: harmonisation): eigenvalue max=' + "{0:.5f}".format(corr_par_eigenval.max())
    file_str = 'harmonisation_eigenvalues_correlation_matrix.png'
    plot_eigenval(corr_par_eigenval, title_str, file_str) 

    title_str = 'Scree plot (correlation matrix: ensemble): eigenvalue max=' + "{0:.5f}".format(corr_ensemble_eigenval.max())
    file_str = 'ensemble_eigenvalues_correlation_matrix.png'
    plot_eigenval(corr_ensemble_eigenval, title_str, file_str) 

    title_str = 'Scree plot (covariance matrix: harmonisation): eigenvalue max=' + "{0:.5f}".format(cov_par_eigenval.max())
    file_str = 'harmonisation_eigenvalues_covariance_matrix.png'
    plot_eigenval(cov_par_eigenval, title_str, file_str) 

    title_str = 'Scree plot (covariance matrix: ensemble): eigenvalue max=' + "{0:.5f}".format(cov_ensemble_eigenval.max())
    file_str = 'ensemble_eigenvalues_covariance_matrix.png'
    plot_eigenval(cov_ensemble_eigenval, title_str, file_str) 

    #
    # Plot Correlation Matrices
    #

    title_str = 'Correlation matrix (harmonisation)'
    file_str = 'harmonisation_correlation_matrix.png'
    plot_eigenvec(corr_par, title_str, file_str) 

    title_str = "Correlation matrix (ensemble)"
    file_str = 'ensemble_correlation_matrix.png'    
    plot_eigenvec(corr_ensemble, title_str, file_str) 

    title_str = "Correlation matrix (harmonisation - ensemble)"
    file_str = 'diff_correlation_matrix.png'
    plot_eigenvec(corr_diff, title_str, file_str) 

    #
    # Plot Covariance Matrices
    #

    title_str = 'Covariance matrix (harmonisation)'
    file_str = 'harmonisation_covariance_matrix.png'
    plot_eigenvec(cov_par, title_str, file_str) 

    title_str = "Covariance matrix (ensemble)"
    file_str = 'ensemble_covariance_matrix.png'    
    plot_eigenvec(cov_ensemble, title_str, file_str) 

    title_str = "Covariance matrix (harmonisation - ensemble)"
    file_str = 'diff_covariance_matrix.png'
    plot_eigenvec(cov_diff, title_str, file_str)     


