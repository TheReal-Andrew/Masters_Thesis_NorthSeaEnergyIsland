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

solutions1  = np.load('v_2030_local_2MAA_1p_solutions.npy')
solutions10 = np.load('v_2030_local_2MAA_10p_solutions.npy')
techs = ['P2X', 'Data']

#%%

gm.solutions_2D(techs, solutions1, n_samples = 10000,
                title = '2D plot of 3D MAA space, mga_slack = 0.01',
                # filename = 'v_2030_local_3MAA_1p_2D_MAA_plot.pdf',
                )

gm.solutions_2D(techs, solutions10, n_samples = 10000,
                title = '2D plot of 3D MAA space, mga_slack = 0.1',
                # filename = 'v_2030_local_3MAA_10p_2D_MAA_plot.pdf',
                )

#%%
from scipy.spatial import ConvexHull

sol1  = solutions1

sol10 = solutions10

x1, y1 = sol1[:,0], sol1[:,1]

x10, y10 = sol10[:,0], sol10[:,1]

fig = plt.figure(figsize = (10,5))

colors = ['tab:blue', 'tab:red', 'tab:purple']

i = 1
for sol in [sol1, sol10]:
    hull = ConvexHull(sol)
    for simplex in hull.simplices:
        plt.plot(sol[simplex, 0], sol[simplex, 1], colors[i])
    i += 1

plt.plot(x1, y1, 'o', label = "Near-optimal, mga = 0.01", color = 'tab:red')
plt.plot(x10, y10, 'o', label = "Near-optimal, mga = 0.1", color = 'tab:purple')



