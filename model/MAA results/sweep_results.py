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
import cdd as pcdd
from scipy.spatial import ConvexHull

tech = 'Data'

sol1 = df_dict[(tech, 0.8)].values
sol2 = df_dict[(tech, 1)].values

intersection = gm.get_intersection(sol1, sol2)

xlim = [0, sol2[:,0].max()]
ylim = [0, sol2[:,1].max()]
zlim = [0, sol2[:,2].max()]

gm.solutions_3D(techs, sol1, markersize = 2, linewidth = 2,
                xlim = xlim, ylim = ylim, zlim = zlim)
gm.solutions_3D(techs, sol2, markersize = 2, linewidth = 2,
                xlim = xlim, ylim = ylim, zlim = zlim)

gm.solutions_3D(techs, intersection, markersize = 2, linewidth = 2,
                xlim = xlim, ylim = ylim, zlim = zlim)

# gm.solutions_2D(techs, sol1,
#                 xlim = [0, 3500], ylim = [0, 6000])
# gm.solutions_2D(techs, sol2,
#                 xlim = [0, 3500], ylim = [0, 6000])
# gm.solutions_2D(techs, intersection,
#                 xlim = [0, 3500], ylim = [0, 6000])

#%%

cmap = plt.get_cmap('viridis')
vmin = 0.5
vmax = 1.5

# plt.figure()
# fig, axs = plt.subplots(2, 2, figsize = (10, 10),)

# Loop through techs
for sensitivity_tech in techs:
    
    plt.figure()
    fig, axs = plt.subplots(2, 2, figsize = (10, 10),)

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
                gm.MAA_density_for_vars(techs, df_dict[(sensitivity_tech, p)].values, [tech0, tech1],
                                        density = False, show_legend = False,
                                        cheb = cheb,
                                        polycolor = cmap(p_norm), ax = ax)


        
