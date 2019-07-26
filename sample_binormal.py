#!/usr/bin/env python

# ipdb> import os; os._exit(1)

# call as: python sample_binormal.py

# =======================================
# Version 0.2
# 25 July, 2019
# michael.taylor AT reading DOT ac DOT uk
# =======================================

import numpy as np
import pandas as pd
from pandas import Series, DataFrame, Panel
import seaborn as sns; sns.set(style="darkgrid")
import matplotlib.pyplot as plt; plt.close("all")

def plot_sample_binormal():
    '''
    # -------------------------------
    # TEST CASE: SAMPLE FROM BINORMAL
    # -------------------------------
    '''

    #
    # Generate random binormal data
    #

    Xmean = np.zeros(2) # [0,0]
    Xcov = np.eye(2) # [[1,0],[0,1]]
    size = 10000
    data1 = np.random.multivariate_normal(Xmean, Xcov, size)
    X1 = data1[:,0]
    X2 = data1[:,1]
    Xmean = [X1.mean(), X2.mean()]
    X = np.stack((X1, X2), axis=0)
    Xcov = np.cov(X)

    #
    # Make 100 independent draws 
    #

    size = 100
    data2 = np.random.multivariate_normal(Xmean, Xcov, size)        
    df1 = pd.DataFrame(data1, columns=['x1', 'x2'])
    df2 = pd.DataFrame(data2, columns=['y1', 'y2'])

    #
    # Plot joint distribution
    #

    # colours = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    #             purple,      blue,      grey,    orange,      navy,     green

    fig, ax = plt.subplots()
    graph = sns.jointplot(x=df1.x1, y=df1.x2, kind="hex", space=0, color="#3498db")
    plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
    cax = graph.fig.add_axes([.81, .15, .02, .5])  # x, y, width, height
    cbar = plt.colorbar(cax=cax)
    cbar.set_label('count')
    graph.x = df2.y1
    graph.y = df2.y2
    graph.plot_joint(plt.scatter, marker="x", color="#e74c3c", s=2)
    graph.x = X1.mean()
    graph.y = X2.mean()
    graph.plot_joint(plt.scatter, marker="x", color="r", s=50)    
    fig.suptitle('2D-sampling from binormal distribution')
    plt.savefig('sampled_binormal.png')    
    
if __name__ == "__main__":

    plot_sample_binormal()


