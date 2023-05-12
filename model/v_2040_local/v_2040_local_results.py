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

year          = 2040
mga_slack     = 0.1   # MAA slack control
study_name    = 'local'
variables     = ['P2X', 'Data', 'Storage']

name  = f'v_{year}_{study_name}_{len(variables)}MAA_{int(mga_slack*100)}p_'
title = f'Model: {study_name}_{year}, for {len(variables)} MAA variables, with {int(mga_slack*100)} % slack'

solutions = np.load(name + 'solutions.npy')
n_opt      = pypsa.Network(f'v_{year}_{study_name}_opt.nc')

# Generate list with optimal values
opt_system = n_opt.generators.loc[n_opt.generators.index.isin(variables)].p_nom_opt.tolist()

opt_system.append(n_opt.stores.loc[n_opt.stores.index.isin(variables)].e_nom_opt['Storage'])

#%%
gm.solutions_2D(variables, solutions, n_samples = 1_000_000,
                title = title,
                opt_system = opt_system,
                xlim = [0, solutions.max()], ylim = [0, solutions.max()],
                filename = f'graphics/v_{year}_{study_name}_{len(variables)}MAA_{int(mga_slack*100)}p_plot_2D_MAA.pdf',
                )

gm.solutions_3D(variables, solutions,
                markersize = 0, linewidth = 0.5,
                filename = f'graphics/v_{year}_{study_name}_{len(variables)}MAA_{int(mga_slack*100)}p_plot_3D_MAA.pdf',
                )


