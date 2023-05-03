# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 09:23:17 2023

@author: lukas
"""

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

year       = 2040
project_name = 'local_nac'
mga_slack  = 0.1

#%%

solutions   = np.load(f'v_{year}_{project_name}_3MAA_10p_solutions.npy')
n_opt       = pypsa.Network(f'v_{year}_{project_name}_opt.nc')
techs       = ['P2X', 'Data', 'Store1']
tech_titles = ['Hydrogen', 'IT', 'Storage']

#%%
# Get optimal values for techs
optimal_solutions = []
for tech in techs:
    if tech == "P2X" or tech == 'Data':
        optimal_value = n_opt.generators.p_nom_opt[tech]
    elif tech == "Store1":
        optimal_value = n_opt.stores.e_nom_opt['Island_store']
        
    optimal_solutions.append(optimal_value)
    
#%% Loop through optimal system and change parameters

# ----- data ---------
tech_df           = tm.get_tech_data(year, 0.07) # Get data

area_use          = tm.get_area_use()
n_opt.area_use      = area_use
n_opt.link_sum_max  = n_opt.generators.p_nom_max['Wind']
n_opt.main_links    = n_opt.links[~n_opt.links.index.str.contains("bus")].index

#%%
# Set up network and load in data

n_i = n_opt.copy()

# ----- data ---------
tech_df           = tm.get_tech_data(year, 0.07) # Get data

area_use          = tm.get_area_use()
n_i.area_use      = area_use
n_i.link_sum_max  = n_i.generators.p_nom_max['Wind']
n_i.main_links    = n_i.links[~n_i.links.index.str.contains("bus")].index
n_i.total_area    = n_opt.total_area 

# ---- Change parameters ------

def sweep_solutions(n_opt, solutions, techs):
    area_use          = tm.get_area_use()
    
    sweeps = []
    
    for tech in ['P2X', 'Data', 'Store1']:
        sweep = np.empty((0, len(techs)), float)
        
        for coeff in np.linspace(0,1,6):
            
            # Reset network
            n_i = n_opt.copy()
    
            solutions_df = pd.DataFrame(solutions, columns = ['P2X', 'Data', 'Store1'])
    
            n_i.area_use      = area_use
            n_i.link_sum_max  = n_i.generators.p_nom_max['Wind']
            n_i.main_links    = n_i.links[~n_i.links.index.str.contains("bus")].index
            n_i.total_area    = n_opt.total_area 
            
            set_value = solutions_df[tech].max() * coeff
            
            if tech == 'P2X' or tech == 'Data':
    
                # Force capacity to be built
                n_i.generators.p_nom_max[tech] = set_value 
                n_i.generators.p_nom_min[tech] = set_value
                
            elif tech == "Store1":
                # Force capacity to be built
                n_i.stores.e_nom_max['Island_store'] = set_value 
                n_i.stores.e_nom_min['Island_store'] = set_value
            
            # ----- Optimization ----- 
            def extra_functionality(n,snapshots):
                # gm.area_constraint(n, snapshots)
                gm.link_constraint(n, snapshots)
                
            n_i.lopf(pyomo = False,
                   solver_name = 'gurobi',
                   # keep_shadowprices = True,
                   # keep_references = True,
                   extra_functionality = extra_functionality,
                   )
            
            values = [n_i.generators.p_nom_opt['P2X'], 
                      n_i.generators.p_nom_opt['Data'],
                      n_i.stores.e_nom_opt['Island_store'],
                      ]
            
            sweep = np.append(sweep, np.array([values]), axis=0)
            
        sweeps.append(sweep)



gm.its_britney_bitch()

#%%

n_i = n_opt.copy()

solutions_df = pd.DataFrame(solutions, columns = ['P2X', 'Data', 'Store1'])

# ----- data ---------
tech_df           = tm.get_tech_data(year, 0.07) # Get data

area_use          = tm.get_area_use()
n_i.area_use      = area_use
n_i.link_sum_max  = n_i.generators.p_nom_max['Wind']
n_i.main_links    = n_i.links[~n_i.links.index.str.contains("bus")].index
n_i.total_area    = n_opt.total_area 

set_value = solutions_df['Store1'].max() * 0

# # Force capacity to be built
# n_i.generators.p_nom_max['Data'] = set_value 
# n_i.generators.p_nom_min['Data'] = set_value

# Force capacity to be built
n_i.stores.e_nom_max['Island_store'] = set_value 
n_i.stores.e_nom_min['Island_store'] = set_value
    
# ----- Optimization ----- 
def extra_functionality(n,snapshots):
    # gm.area_constraint(n, snapshots)
    gm.link_constraint(n, snapshots)
    
n_i.lopf(pyomo = False,
       solver_name = 'gurobi',
       # keep_shadowprices = True,
       # keep_references = True,
       extra_functionality = extra_functionality,
       )

#%%

gm.bake_local_area_pie(n_i, plot_title = 'n_i')

gm.bake_capacity_pie(n_i, plot_title = 'n_i')

# gm.bake_local_area_pie(n_opt, plot_title = 'n_opt')

# gm.bake_capacity_pie(n_opt, plot_title = 'n_opt')

#%%
gm.solutions_2D(techs, solutions, n_samples = 200000,
                tech_titles = tech_titles,
                bins = 35,
                optimal_solutions = optimal_solutions,
                title = f'MAA results - {year}, mga_slack = {int(mga_slack)*100}% \n No area constraint',
                filename = f'graphics/v_{year}_{project_name}_{len(techs)}MAA_{mga_slack*100}p_plot_2D_MAA.pdf'
                )

#%%
gm.solutions_3D(techs, solutions, 
                figsize = (20,20),
                )

#%%

from gorm import sample_in_hull, set_plot_options

n_samples = 200000

opt_system = optimal_solutions

bins = 35

title = 'yolo'
filename = None

# Take a multi-dimensional MAA polyhedron, and plot each "side" in 2D.
# Plot the polyhedron shape, samples within and correlations.
import pandas
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

pad = 5
ncols = len(techs) if opt_system == None else len(techs)+1

if tech_titles == None: 
    tech_titles = techs

# Sample polyhedron
d = sample_in_hull(solutions, n_samples)

# -------- create correlation matrix --------------------------
# Create dataframe from samples
d_df = pandas.DataFrame(d, columns = techs)

# Calculate correlation and normalize
d_corr = d_df.corr()

# Calculate normalized correlation, used to color heatmap.
d_temp = d_corr + abs(d_corr.min().min())
d_norm = d_temp / d_temp.max().max()

# -------- Set up plot ----------------------------------------
set_plot_options()

text_lift = 1.075

# define the endpoints of the colormap
red    = (1.0, 0.7, 0.6)  # light red
yellow = (1.0, 1.0, 0.8)  # light yellow
green  = (0.6, 1.0, 0.6)  # light green

# define the colormap
cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', [red, yellow, green])

# Initialize and adjust figure
plt.figure()
fig, axs = plt.subplots(len(techs), len(techs), figsize = (20,15))
fig.subplots_adjust(wspace = 0.4, hspace = 0.4)

# Set titles
for ax, col in zip(axs[0], tech_titles):
    ax.set_title(col + '\n')

for ax, row in zip(axs[:,0], tech_titles):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords = ax.yaxis.label, textcoords='offset points',
                size = 24, ha = 'right', va = 'center',
                rotation = 90)

# -------- Plotting -------------------------------

# Upper triangle of subplots
for i in range(0, len(techs)):
    for j in range(0, i):
        
        corr = d_norm[techs[i]][techs[j]] # Is only used for coloring
        num  = d_corr[techs[i]][techs[j]] # Is shown
        
        ax = axs[j][i]
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Write correlation
        corr_text = str(round(num,2))
        ax.text(0.5, 0.5, corr_text, ha='center', va='center', fontsize=20)
        
        ax.text(0.5, text_lift, 'Correlation', ha='center', va='top',
                transform=ax.transAxes, fontsize = 16, color = 'gray')
        
        # Change bg color according to correlation
        ax.patch.set_facecolor(cmap(corr))


# Diagonal plotting
for j in range(0, len(techs)):
    
    ax = axs[j][j]
    
    d_df[techs[j]].hist(bins = 50, ax = ax,
                        color = 'tab:purple', rwidth = 0.9,
                        label = 'histogram')
    
    ax.text(0.5, text_lift, 'Histogram', ha='center', va='top', 
            transform=ax.transAxes, fontsize = 16, color = 'gray')


# lower traingle of subplots
for j in range(0, len(techs)):
    for i in range(0, j):
        
        ax = axs[j][i]
        
        ax.text(0.5, text_lift, 'Scatter plot with vertices', ha='center', va='top',
                transform=ax.transAxes, fontsize=16, color = 'gray')
        
        # MAA solutions
        x, y = solutions[:,i],   solutions[:,j]
        
        # Set x and y as samples for this dimension
        x_samples = d[:,i]
        y_samples = d[:,j]
        
        # --------  Create 2D histogram --------------------
        hist, xedges, yedges = np.histogram2d(x_samples, y_samples,
                                              bins = bins)

        # Create grid for pcolormesh
        X, Y = np.meshgrid(xedges, yedges)
        
        # Create pcolormesh plot with square bins
        ax.pcolormesh(X, Y, hist.T, cmap = 'Blues', zorder = 0)
        
        # Create patch to serve as hexbin label
        hb = mpatches.Patch(color = 'tab:blue')
        
        ax.grid('on')
        
        # --------  Plot hull --------------------
        hull = ConvexHull(solutions[:,[i,j]])
        
        # plot simplexes
        for simplex in hull.simplices:
            l0, = ax.plot(solutions[simplex, i], solutions[simplex, j], 'k-', 
                    color = 'silver', label = 'faces', zorder = 0)
            
        # # Plot vertices from solutions
        # l1, = ax.plot(x, y,
        #           'o', label = "Near-optimal",
        #           color = 'lightcoral', zorder = 2)
        
        # list of legend handles and labels
        l_list, l_labels   = [l0, hb], ['Polyhedron face', 'Sample density']
        
        # l_list, l_labels   = [], []
        
        # optimal solutions
        if not opt_system == None:
            x_opt, y_opt = opt_system[i],   opt_system[j]
            
            ax.plot(sweeps[0][:,i], sweeps[0][:,j], 
                    color = 'lightcoral',
                    marker = 'o', linestyle = '-')
            
            ax.plot(sweeps[1][:,i], sweeps[1][:,j], 
                    color = 'gold',
                    marker = 'o', linestyle = '-')
            
            ax.plot(sweeps[2][:,i], sweeps[2][:,j], 
                    color = 'palegreen',
                    marker = 'o', linestyle = '-')
            
            # Plot optimal solutions
            l2, = ax.plot(x_opt, y_opt,
                      'o', label = "Optimal", 
                      ms = 15, color = 'red',
                      zorder = 3)
            
            l_list.append(l2)
            l_labels.append('Optimal solution')

# Place legend below subplots
ax = axs[len(techs)-1, int(np.median([1,2,3]))-1] # Get center axis
ax.legend(l_list,
          l_labels, 
          loc = 'center', ncol = ncols,
          bbox_to_anchor=(0.5, -0.25),fancybox=False, shadow=False,)

fig.suptitle(title, fontsize = 24)

if not filename == None:
    fig.savefig(filename, format = 'pdf', bbox_inches='tight')













