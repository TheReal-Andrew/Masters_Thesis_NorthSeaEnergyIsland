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

solutions = np.load('v_2040_local_3MAA_10p_solutions.npy')
techs = ['P2X', 'Data', 'Store1']

gm.solutions_2D(techs, solutions, n_samples = 10000,
                title = '2D plot of 3D MAA space',
                filename = 'v_2030_local_3MAA_10p_2D_MAA_plot.pdf')





