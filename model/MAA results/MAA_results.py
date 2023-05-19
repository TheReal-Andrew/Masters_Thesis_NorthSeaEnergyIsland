# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:00:12 2023

@author: lukas
"""

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

import gorm as gm
import tim as tm
import pypsa_diagrams as pdiag

gm.set_plot_options()

#%% Load solutions

project = 'local'
n_MAA   = 3

# solutions = np.load('../v_2030_links_G2/v_2030_links_G2_3MAA_10p_solutions.npy')
# techs     = ['DE', 'NL', 'GB']

# solutions = np.load('../v_2030_links_G3/v_2030_links_G3_3MAA_10p_solutions.npy')
# techs     = ['DK', 'NO', 'BE']

solutions30 = np.load(f'../v_2030_{project}/v_2030_{project}_{n_MAA}MAA_10p_solutions.npy')
solutions40 = np.load(f'../v_2040_{project}/v_2040_{project}_{n_MAA}MAA_10p_solutions.npy')
techs     = ['P2X', 'Data', 'Storage']

n_samples    = 100000
bins         = 50
density_vars = ['P2X', 'Data']
colors       = gm.get_color_codes()
opt_system = None

from gorm import sample_in_hull

#%% Overlapping histograms

n_samples   = 100_000
tech_colors = gm.get_color_codes()

# project, n_MAA, techs, title

projects    = ['local', 'links_G2', 'links_G3']
techs_list  = [['P2X', 'Data', 'Storage'],
               # ['P2X', 'Data', 'Storage'],
               ['DE', 'NL', 'GB'],
               ['DK', 'NO', 'BE']
               ]

for techs, project in zip(techs_list, projects):
    #Local:
    filename = f'MAA_overlap_hist_{project}.pdf'
    n_MAA   = 3
    sol30 = np.load(f'../v_2030_{project}/v_2030_{project}_{n_MAA}MAA_10p_solutions.npy')
    sol40 = np.load(f'../v_2040_{project}/v_2040_{project}_{n_MAA}MAA_10p_solutions.npy')
    sols  = [sol30, sol40]
    title = f'Histograms for project: {project}'
    gm.histograms_3MAA(techs, sols, title = title, n_samples = n_samples,
                       filename = filename,)


#%%
#Local:
project = 'local'
n_MAA   = 3
solutions30 = np.load(f'../v_2030_{project}/v_2030_{project}_{n_MAA}MAA_10p_solutions.npy')
solutions40 = np.load(f'../v_2040_{project}/v_2040_{project}_{n_MAA}MAA_10p_solutions.npy')

solutions_list = [solutions30,
                   solutions40,
                  ]
    
gm.histograms_3MAA(techs, solutions_list, n_samples = n_samples)

#%% Histogram + MAA density function
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull

fig, axs = plt.subplots(1, 2, figsize = (20,5),
                       gridspec_kw={'width_ratios': [1, 3]},
                      )

for solutions in [solutions30, solutions40]:
    # Solutions to df
    solutions_df = pd.DataFrame(data = solutions,
                                columns = techs)
    
    # Sampling
    samples = sample_in_hull(solutions, n_samples)
    samples_df = pd.DataFrame(samples, columns = techs)
    
    #
    
    # ----------------------- Histograms plot -----------------------------
    for tech in techs:
            
        ax = axs[1]
        
        sns.histplot(samples_df[tech].values, 
                     line_kws = {'linewidth':3},
                     color = colors[tech],
                     alpha = 1/len(techs*2),
                     kde = True,
                     ax = ax, label = tech,
                     )
            
    axs[1].legend(bbox_to_anchor=(0.5, -0.25), loc = 'center', 
                  ncol = len(techs), fancybox=False, shadow=False,)
    axs[1].set(xlabel = 'Installed capacity [MW]/[MWh]', ylabel = 'Frequency')
    
    #%
    # MAA density plot
    
    x, y = solutions_df[density_vars[0]], solutions_df[density_vars[1]]
    x_samples, y_samples = samples_df[density_vars[0]], samples_df[density_vars[1]]
    
    # --------  Create 2D histogram --------------------
    hist, xedges, yedges = np.histogram2d(x_samples, y_samples,
                                          bins = bins)
    
    # Create grid for pcolormesh
    X, Y = np.meshgrid(xedges, yedges)
    
    # Create pcolormesh plot with square bins
    axs[0].pcolormesh(X, Y, hist.T, cmap = 'Blues', zorder = 0)
    
    # Create patch to serve as hexbin label
    hb = mpatches.Patch(color = 'tab:blue')
    
    # Set labels and grid
    xlabel = 'IT' if density_vars[0] == 'Data' else density_vars[0] 
    ylabel = 'IT' if density_vars[1] == 'Data' else density_vars[1]
    xUnit  = ' [MWh]' if density_vars[0] == 'Storage' else ' [MW]'
    yUnit  = ' [MWh]' if density_vars[1] == 'Storage' else ' [MW]'
    
    axs[0].grid('on')
    axs[0].set(xlabel = xlabel + xUnit, ylabel = ylabel + yUnit)
    
    # --------  Plot hull --------------------
    hull = ConvexHull(solutions_df[density_vars])
    
    solutions_plot = solutions_df[density_vars].values
    
    # plot simplexes
    for simplex in hull.simplices:
        l0, = axs[0].plot(solutions_df[density_vars[0]][simplex],
                          solutions_df[density_vars[1]][simplex],
                          '-', color = 'silver', label = 'faces', zorder = 0)
        
    l_list, l_labels   = [l0, hb], ['Polyhedron face', 'Sample density']
        
    # # optimal solutions
    # if not opt_system == None:
    #     x_opt, y_opt = opt_system[i],   opt_system[j]
        
    #     # Plot optimal solutions
    #     l2, = ax.plot(x_opt, y_opt,
    #               'o', label = "Optimal", 
    #               ms = 20, color = 'red',
    #               zorder = 3)
        
    #     l_list.append(l2)
    #     l_labels.append('Optimal solution')
        
        
    ncols = len(techs) if opt_system == None else len(techs)+1
    axs[0].legend(l_list, l_labels, 
                  bbox_to_anchor=(0.5, -0.25),
                  loc = 'center', ncol = ncols,fancybox=False, shadow=False,)


#%%



















