# -*- coding: utf-8 -*-
"""
Created on Wed May 24 12:08:58 2023

@author: lukas
"""
#%% Modules
import os
import sys
# Add modules folder to path
os.chdir(os.path.join(os.path.dirname(__file__))) # Change working directory
sys.path.append(os.path.abspath('../../modules')) # Add modules to path

import pypsa
import pandas as pd
import numpy as np
from ttictoc import tic, toc
import matplotlib.pyplot as plt
from pywaffle import Waffle
import polytope

import gorm as gm
import tim as tm
import pypsa_diagrams as pdiag

gm.set_plot_options()

colors = gm.get_color_codes()

#%% Import data

techs = ['P2X', 'Data', 'Storage']

constant_name = 'v_2030_sensitivity_sweep_3MAA_10p_sweep'

df_dict = {}

for tech in techs:
    
    for p in np.linspace(0.5, 1.5, 11):
        
        sol = np.load(f'sweep_results/{constant_name}_{tech}_{round(p,1)}_solutions.npy')
        
        sol_df = pd.DataFrame(sol, columns = techs)
        
        # Define dict keys
        name   = tech
        number = round(p, 1)
        
        key = (name, number)
        
        df_dict[key] = sol_df
        

#%% 

cmap = plt.get_cmap('viridis')
vmin = 0.5
vmax = 1.5

# Loop through techs
for sensitivity_tech in techs:
    
    plt.figure()
    fig, axs = plt.subplots(2, 2, figsize = (10, 10),)
    
    axs[0,1].set_visible(False)
    
    fig.suptitle(f'MAA sensitivity analysis of {sensitivity_tech} capital cost',
                 y = 0.925)

    # Loop through axes
    for j in range(len(techs)-1):
        for i in range(j+1):
            
            ax = axs[j][i]
            tech0 = techs[i]
            tech1 = techs[j+1]
            
            # loop through percentages
            for p in np.linspace(0.5, 1.5, 11):
            
                cheb = False
                if p == 0.5: # and sensitivity_tech == 'Data':
                    cheb = True
                    
                p = round(p, 1)
                p_norm = (p - vmin) / (vmax - vmin)
                
                color = 'tab:red' if p == 0.5 else cmap(p_norm) 
                zorder = 1 if p == 0.5 else 0
                
                gm.MAA_density_for_vars(techs, df_dict[(sensitivity_tech, p)].values, [tech0, tech1],
                                        density = False, show_legend = False,
                                        cheb = cheb, zorder = zorder,
                                        polycolor = color, ax = ax)


#%% Minimum areas to overlap

# Get solutions
P2X_min   = df_dict[('P2X', 1.5)]
IT_min    = df_dict[('Data', 0.5)]
Store_min = df_dict[('Storage', 1.5)]

# New plot
plt.figure()
fig, axs = plt.subplots(2, 2, figsize = (10, 10))
axs[0,1].set_visible(False)

for j in range(len(techs)-1):
    for i in range(j+1):
        
        ax = axs[j][i]
        tech0 = techs[i]
        tech1 = techs[j+1]
        
        k = 0
        for sol in [P2X_min, IT_min, Store_min]:
            gm.MAA_density_for_vars(techs, sol.values, [tech0, tech1],
                                    density = False, show_legend = False,
                                    ax = ax, polycolor = colors[techs[k]]
                                    )
            
            k += 1

        
