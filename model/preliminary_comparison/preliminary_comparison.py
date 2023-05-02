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
from scipy.spatial import ConvexHull

gm.set_plot_options()

#%% Import optimal systems with and without constraints.

# Load MAA solutions
techs = techs = ['P2X', 'Data', 'Store1']
solutions_2030 = np.load('../v_2030_preliminary/OLD_results_2023_01_05/v_2030_preliminary_3MAA_10p_solutions.npy')
solutions_2030_nac = np.load('../v_2030_preliminary_nac/v_2030_preliminary_nac_3MAA_10p_solutions.npy')
solutions_2040 = np.load('../v_2040_preliminary/v_2040_preliminary_3MAA_10p_solutions.npy')

# Load 2030 optimal networks
n_2030     = pypsa.Network('../v_2030_preliminary/OLD_results_2023_01_05/OLD_WIND__v_2030_preliminary_opt.nc')
n_2030_nac = pypsa.Network('../v_2030_preliminary_nac/v_2030_preliminary_nac_opt.nc')

#Load 2040 optimal networks
n_2040     = pypsa.Network('../v_2040_preliminary/v_2040_preliminary_opt.nc')
n_2040_nac = pypsa.Network('../v_2040_preliminary_nac/v_2040_preliminary_nac_opt.nc')

networks = [n_2030, n_2030_nac, n_2040, n_2040_nac]

export = pd.DataFrame( [
                        [n_2030.objective, n_2040.objective],
                        [n_2030_nac.objective, n_2040_nac.objective]
                        ],
                      columns = ['2030', '2040'],
                      index   = ['With constraint', 'Without constraint']
                      )

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

#%% Overlapping polyhedrons

import matplotlib.patches as mpatches

sol_list    = [
                solutions_2030, 
                solutions_2040,
                # solutions_2030_nac
               ]
colors_list = [
                'tab:blue', 
                'tab:purple'
               ]

plt.figure()
fig, axs = plt.subplots(len(techs), len(techs), figsize = (20,15))
fig.subplots_adjust(wspace = 0.4, hspace = 0.4)
pad = 5

# Set titles
for ax, col in zip(axs[0], techs):
    ax.set_title(col + '\n')

for ax, row in zip(axs[:,0], techs):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')

for sol, color in zip(sol_list, colors_list):

    for j in range(0, len(techs)):
        for i in range(0, j):
            
            ax = axs[j][i]
            
            x = sol[:,i]
            y = sol[:,j]
            
            hull = ConvexHull(sol[:,[i,j]])
            
            # plot simplexes
            for simplex in hull.simplices:
                l0, = ax.plot(sol[simplex, i], sol[simplex, j], 'k-', 
                        label = 'faces', zorder = 0)
                
            # Plot vertices from solutions
            l1, = ax.plot(x, y,
                      'o', label = "Near-optimal", 
                      color = color,
                      zorder = 2, alpha = 0.25)
            
            hull_poly = plt.Polygon(sol[:,[i,j]][hull.vertices], 
                                    alpha = 0.5, color = color)
            ax.add_patch(hull_poly)
            
            # Create patch to serve as hexbin label
    
p1 = mpatches.Patch(color = colors_list[0])
p2 = mpatches.Patch(color = colors_list[1])
            
ax = axs[len(techs)-1, int(np.median([1,2,3]))-1] # Get center axis
ax.legend([p1, p2], ['2030', '2040'], 
          loc = 'center', ncol = 3,
          bbox_to_anchor=(0.5, -0.45),fancybox=False, shadow=False,)

fig.suptitle('MAA areas for 2030 and 2040, mga_slack = 0.1', fontsize = 24)

