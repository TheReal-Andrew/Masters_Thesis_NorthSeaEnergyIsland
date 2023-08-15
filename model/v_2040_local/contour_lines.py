# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:46:54 2023

@author: lukas
"""

#%% Packages

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
from matplotlib.lines import Line2D
import polytope as pc
import matplotlib.patches as mpatches

#%% Import

year          = 2040
study_name    = 'local'
variables     = ['P2X', 'Data', 'Storage']


sol10 = np.load('v_2040_local_3MAA_10p_solutions.npy')/1000
sol5  = np.load('v_2040_local_3MAA_5p_solutions.npy')/1000
sol1  = np.load('v_2040_local_3MAA_1p_solutions.npy')/1000
n_opt = pypsa.Network(f'v_{year}_{study_name}_opt.nc')


opt = n_opt.generators.loc[n_opt.generators.index.isin(variables)].p_nom_opt.tolist()
opt.append(n_opt.stores.loc[n_opt.stores.index.isin(variables)].e_nom_opt['Storage'])

#
opt = [x/1000 for x in opt]

#%% Set up plot

n_samples = 1000_000
legend_ncols = 3
c10 = 'tomato'
c5  = 'gold'
c1  = 'yellowgreen'
linewidth = 3

plt.figure()

fig, axs = plt.subplots(1, 4, figsize = (24,5),)
fig.subplots_adjust(wspace = 0.2, hspace = 0.2)
fig.suptitle('2040, local demand, IT/Storage - Near-optimal spaces for 10%, 5% and 1% slack'
             , fontsize = 24, y = 1.05)

axs[0].set_title('Near-optimal spaces overlap')
axs[1].set_title('10% slack')
axs[2].set_title('5% slack')
axs[3].set_title('1% slack')

# Calculate cheb. center and radius -------------------------------------------

cr10 = pc.qhull(sol10).chebR
cr5  = pc.qhull(sol5).chebR
cr1  = pc.qhull(sol1).chebR

# Contour line plots ----------------------------------------------------------
# Plot 10%
gm.MAA_density_for_vars(variables, sol10, ['Data', 'Storage'],
                             density = False, cheb = True,
                             ax = axs[0],
                             polycolor = c10,
                             show_legend = False,
                             linewidth = linewidth,
                             cheb_color = c10,
                             )

# Plot 5%
gm.MAA_density_for_vars(variables, sol5, ['Data', 'Storage'],
                             density = False, cheb = True,
                             ax = axs[0],
                             polycolor = c5,
                             show_legend = False,
                             linewidth = linewidth,
                             cheb_color = c5,
                             )

# Plot 1%
gm.MAA_density_for_vars(variables, sol1, ['Data', 'Storage'],
                             density = False, cheb = True,
                             ax = axs[0], opt_system = opt,
                             polycolor = c1,
                             show_legend = False,
                             linewidth = linewidth,
                             cheb_color = c1,
                             )

# Individual plots ------------------------------------------------------------
gm.MAA_density_for_vars(variables, sol10, ['Data', 'Storage'],
                             density = True, cheb = True,
                             show_legend = False,
                             opt_system = opt,
                             n_samples = n_samples,
                             ax = axs[1], ncols = legend_ncols,
                             legend_v = -0.475,
                             polycolor = c10,
                             cheb_color = c10,
                             linewidth = linewidth,
                             )

gm.MAA_density_for_vars(variables, sol5, ['Data', 'Storage'],
                             density = True, cheb = True,
                             show_legend = False,
                             opt_system = opt,
                             n_samples = n_samples,
                             ax = axs[2], ncols = legend_ncols,
                             legend_v = -0.475,
                             polycolor = c5,
                             cheb_color = c5,
                             linewidth = linewidth,
                             )

gm.MAA_density_for_vars(variables, sol1, ['Data', 'Storage'],
                             density = True, cheb = True,
                             show_legend = False,
                             opt_system = opt,
                             n_samples = n_samples,
                             ax = axs[3], ncols = legend_ncols,
                             legend_v = -0.475,
                             polycolor = c1,
                             cheb_color = c1,
                             linewidth = linewidth,
                             )

# Build legend ---------------------------------------------------------------

# Boundary line legend entries-----------
l10 = Line2D([0], [0], marker = '', 
              ms = 0, linestyle = '-', 
              linewidth = linewidth,
              color = c10
              )

l5 = Line2D([0], [0], marker = '', 
              ms = 0, linestyle = '-', 
              linewidth = linewidth,
              color = c5
              )

l1 = Line2D([0], [0], marker = '', 
              ms = 0, linestyle = '-', 
              linewidth = linewidth,
              color = c1
              )

# Cheb. center legend entries------------
p10 = Line2D([0], [0], marker = 'o', 
              ms = 15, linestyle = '', 
              color = c10
              )

p5 = Line2D([0], [0], marker = 'o', 
              ms = 15, linestyle = '', 
              color = c5
              )

p1 = Line2D([0], [0], marker = 'o', 
              ms = 15, linestyle = '', 
              color = c1
              )

# Density legend entry ------------------
hb = mpatches.Patch(color = 'tab:blue')

# Optimal point legend entry
opt_legend = Line2D([0], [0], marker = '*', color = 'gold',
            markeredgecolor = 'darkorange', markeredgewidth = 2,
            markersize = 25, label = 'Optimal Solutions',
            linestyle = '',)

handles = [l10, l5, l1, p10, p5, p1, hb, opt_legend]
labels  = ['10% slack', 
           '5% slack', 
           '1% slack', 
           f'10% Cheb. center (r = {round(cr10,3)})',
           f'5% Cheb. center (r = {round(cr5,3)})',
           f'1% Cheb. center (r = {round(cr1,3)})',
           'Sample density', 'Optimal solution'
           ] 

axs[0].legend(handles, labels, loc = 'lower center',
              ncols = legend_ncols,
               bbox_to_anchor=(2, -0.475),
              )

filename = '2040_local_IT-Storage_contour_lines.pdf'
fig.savefig(f'graphics/{filename}', format = 'pdf', bbox_inches='tight')

























