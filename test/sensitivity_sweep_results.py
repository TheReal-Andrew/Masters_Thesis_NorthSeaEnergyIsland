# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 13:19:05 2023

@author: lukas
"""

#%% Import

import os
import sys
# Add modules folder to path
os.chdir(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.abspath('../../modules')) 

import pypsa
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

import gorm as gm
import tim as tm
from ttictoc import tic,toc
gm.set_plot_options()

#%% Results

year          = 2030
mga_slack     = 0.1   # MAA slack control
study_name    = 'sensitivity_sweep'
variables     = ['P2X', 'Data']
percent       = [0.6, 0.8, 1, 1.2]

name  = f'v_{year}_{study_name}_{len(variables)}MAA_{int(mga_slack*100)}p_'
title = f'Model: {study_name}_{year}, for {len(variables)} MAA variables, with {int(mga_slack*100)} % slack'

solutions = np.load(name + f'sweep_{variables[0]}_{percent[0]}_'+'solutions.npy')

#%%

# fill_polyhedron(solutions, ax = ax)

# gm.MAA_density(variables, sol_t0_p0_df)

# fig, ax = plt.subplots(1, 1, figsize = (8,8))

colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:purple']

for tech in variables:
    
    fig, ax = plt.subplots(1, 1, figsize = (8,8),
                           )
    
    i = 0
    for p in percent:
        
        color = colors[i]
        
        if tech == 'Data' and p == 1.2:
            continue
        
        solutions = np.load(name + f'sweep_{tech}_{p}_'+'solutions.npy')
        
        solutions = solutions + np.random.random() * 100
        
        gm.fill_polyhedron(variables, solutions, 
                           ax = ax, fillcolor = color,
                           label = f'{tech} {p}')
        
        i += 1
        
    # Add a legend
    ax.legend()
    
        



