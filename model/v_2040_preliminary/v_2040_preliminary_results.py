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

#%%

solutions2 = np.load('MAA_solutions.npy')
techs2 = ['P2X', 'Data', 'Store1']

d = gm.sample_in_hull(solutions2, n = 10000)

# d_df = pd.DataFrame(d,
#                     columns = techs2)

# ds = d_df.sort_values('Data')

# ds2 = d_df.sort_values('P2X')

# ds3 = d_df.sort_values('Store1')

# gm.solutions_2D(techs, solutions, n_samples = 1000)

#%%

d_corr = d_df.corr()

