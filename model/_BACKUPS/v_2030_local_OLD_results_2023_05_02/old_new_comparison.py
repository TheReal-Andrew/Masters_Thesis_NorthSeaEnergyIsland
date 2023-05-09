#%% -*- coding: utf-8 -*-
"""
Created on Mon May  1 09:36:37 2023

@author: lukas
"""

import os
import sys
# Add modules folder to path
os.chdir(os.path.join(os.path.dirname(__file__))) # Change working directory
sys.path.append(os.path.abspath('../../modules')) # Add modules to path

import pypsa
import pandas as pd

import gorm as gm
import tim as tm
import pypsa_diagrams as pdiag

gm.set_plot_options()

#%% 

n_new = pypsa.Network('OLD_WIND__v_2030_preliminary_opt.nc')
n_old = pypsa.Network('v_2030_preliminary_opt.nc')