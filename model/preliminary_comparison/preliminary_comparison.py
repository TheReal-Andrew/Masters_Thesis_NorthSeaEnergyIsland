# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 09:23:17 2023

@author: lukas
"""
#%% import
import os
import sys
# Add modules folder to path
os.chdir(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.abspath('../../modules')) 

import pypsa
from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints, get_dual, get_con, write_objective, get_sol, define_variables
import pandas as pd
import numpy as np

import gorm as gm
import tim as tm
import pypsa_diagrams as pdiag
import matplotlib.pyplot as plt

gm.set_plot_options()

#%% Import optimal systems with and without constraints.

# Load 2030 optimal networks
n_2030     = pypsa.Network('../v_2030_preliminary/v_2030_preliminary_opt.nc')
n_2030_nac = pypsa.Network('../v_2030_preliminary_nac/v_2030_preliminary_nac_opt.nc')

#Load 2040 optimal networks
n_2040     = pypsa.Network('../v_2040_preliminary/v_2040_preliminary_opt.nc')
n_2040_nac = pypsa.Network('../v_2040_preliminary_nac/v_2040_preliminary_nac_opt.nc')

networks = [n_2030, n_2030_nac, n_2040, n_2040_nac]

#%% Compare optimal systems

# Create figure and ravel axes for iteration
fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (10,10))
axs = axs.ravel()

# Get data and put into lists to iterate through
area = tm.get_area_use()
titles   = ['2030', '2030, no area constraint', '2040', '2040, no area constraint']

# Loop through axes and plot piechart on each.
i = 0
for n, title in zip(networks, titles): 
    n.area_use = area # Set area use coefficients in network
    gm.bake_local_area_pie(n, title, 
                            ax = axs[i]
                           ) # Create piechart of area use in solution
    i += 1
   
# Add legend in bottom of subplots
labels =  ["P2X", "Data", "Store"]
plt.legend(labels = labels, 
          loc = 'center',
          ncol = 3,
          bbox_to_anchor=(-0.1, -0.1), fancybox = False, shadow = False,)

# Create overall title for subplots
plt.suptitle('Optimal area use', fontsize = 24)

# Save figure
# fig.savefig('Optimal_area_use_piecharts.pdf', format = 'pdf', bbox_inches='tight')

#%% Capacity pies
fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (10,10))
axs = axs.ravel()

for n in networks: n.main_links = n.links[~n.links.index.str.contains("bus")].index

titles   = ['2030', '2030, no area constraint', '2040', '2040, no area constraint']

# Loop through axes and plot piechart on each.
i = 0
for n, title in zip(networks, titles): 
    
    gm.bake_capacity_pie(n, title, 
                         ax = axs[i]
                         ) # Create piechart of area use in solution
    i += 1
    
# Add legend in bottom of subplots
labels =  ["P2X", "Data", "Store", "Links"]
plt.legend(labels = labels, 
          loc = 'center',
          ncol = 4,
          bbox_to_anchor=(-0.1, -0.1), fancybox = False, shadow = False,)

# Create overall title for subplots
plt.suptitle('Optimal capacities', fontsize = 24)

# Save figure
# fig.savefig('Optimal_capacities_piecharts.pdf', format = 'pdf', bbox_inches='tight')