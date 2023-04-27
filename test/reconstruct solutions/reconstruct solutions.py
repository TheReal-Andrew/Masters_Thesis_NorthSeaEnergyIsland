# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 13:42:21 2023

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

#%% reconstruct solutions from saved networks
solutions = []

for i in range(1,7):
    
    n = pypsa.Network('v_2030_preliminary_3MAA_10p_' + str(i) + '.nc')

    gens = n.generators.p_nom_opt
    stores = n.stores.e_nom_opt

    MAA_point = [gens['P2X'], gens['Data'], stores['Island_store']]
    
    solutions.append(MAA_point)
    
solutions = np.asarray(solutions)

MAA_solutions     = f'v_2030_preliminary_3MAA_10p_solutions.npy'

np.save(MAA_solutions, solutions)