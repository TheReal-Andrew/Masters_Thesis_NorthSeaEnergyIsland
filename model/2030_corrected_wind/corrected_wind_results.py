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

#%% optimal results

n30 = pypsa.Network('v_2030_corrected_wind_opt.nc')
n40 = pypsa.Network('v_2040_corrected_wind_opt.nc')

n = n40

# Define links that go to and from island
main_links          = n.links.loc[n.links.bus0 == "Energy Island"].index
main_links2          = n.links.loc[n.links.bus1 == "Energy Island"].index

# Time series of island to country link, positive when island is sending
n.links_t.p0[main_links].plot.area(figsize = (20,10), title = 'Corrected: Positive if island is sending')

# Time series of country to island link, negative when island is recieving
n.links_t.p1[main_links2].plot.area(figsize = (20,10), title = 'Corrected: Negative if island is recieving')

#%% Results

# year          = 2030
# mga_slack     = 0.1   # MAA slack control
# study_name    = 'base0'
# variables     = ['P2X', 'Data', 'Store1']

# name  = f'v_{year}_{study_name}_{len(variables)}MAA_nac_{int(mga_slack*100)}p_'
# title = f'Model: {study_name}_{year}, for {len(variables)} MAA variables, with {int(mga_slack*100)} % slack'

# solutions = np.load(name + 'solutions.npy')

# gm.solutions_2D(variables, solutions, n_samples = 1000,
#                 title = title,
#                 filename = 'v_{year}_{study_name}_{len(variables)}MAA_{int(mga_slack*100)}p_plot_2D_MAA.pdf'
#                 )


