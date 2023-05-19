# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 14:43:37 2023

@author: Linda
"""

import os
import sys
# Add modules folder to path
sys.path.append(os.path.abspath('../../modules')) 

import pypsa
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import pandas

import gorm as gm
import tim as tm
from ttictoc import tic,toc
gm.set_plot_options()

#%% Control

year = 2030
mga_slack     = 0.1   # 10%

input_name = 'base0_opt.nc'

#%% Set up network and load in data

n = pypsa.Network(input_name) #Load network from netcdf file

n.area_use      = tm.get_area_use()
n.link_sum_max  = n.generators.p_nom_max['Wind']
n.main_links    = n.links.loc[n.links.bus0 == "Energy Island"].index

 #%% Optimization
 
def extra_functionality(n,snapshots):
    gm.area_constraint(n, snapshots)
    gm.link_constraint(n, snapshots)
    gm.marry_links(n, snapshots)
    
n.lopf(pyomo = False,
       solver_name = 'gurobi',
       keep_shadowprices = True,
       keep_references = True,
       extra_functionality = extra_functionality,
       )

