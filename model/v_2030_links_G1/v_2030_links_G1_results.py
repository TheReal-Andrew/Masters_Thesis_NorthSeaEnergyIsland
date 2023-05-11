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
study_name    = 'links_G1'
variables     = ['DK', 'DE', 'NL', 'BE']

name  = f'v_{year}_{study_name}_{len(variables)}MAA_{int(mga_slack*100)}p_'
title = f'Model: {study_name}_{year}, for {len(variables)} MAA variables, with {int(mga_slack*100)} % slack'

solutions = np.load(name + 'solutions.npy')
n_opt     = pypsa.Network(f'v_{year}_{study_name}_opt.nc')

bus_df = tm.get_bus_df()

# filter the rows based on the abbreviations in the 'variables' list
countries  = bus_df.loc[bus_df['Abbreviation'].isin(variables)]['Bus name'].tolist()
opt_system = n_opt.links[n_opt.links['bus0'].isin(countries)].p_nom_opt.tolist()


#%%
gm.solutions_2D(variables, solutions, n_samples = 1_000_000,
                title = title,
                opt_system = opt_system,
                filename = f'graphics/v_{year}_{study_name}_{len(variables)}MAA_{int(mga_slack*100)}p_plot_2D_MAA.pdf',
                )


