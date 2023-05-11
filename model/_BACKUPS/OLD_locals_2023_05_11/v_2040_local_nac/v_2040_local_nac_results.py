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

year         = 2040
project_name = 'local_nac'
mga_slack    = 0.1

#%%

solutions   = np.load(f'v_{year}_{project_name}_3MAA_10p_solutions.npy')
n_opt       = pypsa.Network(f'v_{year}_{project_name}_opt.nc')
techs       = ['P2X', 'Data', 'Store1']
tech_titles = ['Hydrogen', 'IT', 'Storage']

solutions_df = pd.DataFrame(solutions,
                            columns = techs)

#%% Full solutions
gm.solutions_2D(techs, solutions, 
                n_samples = 100000, bins = 35,
                xlim = [0, None], ylim = [0, None],
                tech_titles = tech_titles,
                title = f'MAA results - {year}, mga_slack = {int(mga_slack)*100}% \n No area constraint',
                # filename = f'graphics/v_{year}_{project_name}_{len(techs)}MAA_{mga_slack*100}p_plot_2D_MAA.pdf'
                )

#%% Single MAA area

tech = ['Data', 'Store1']

x = n_opt.generators.p_nom_opt['Data']
y = n_opt.stores.e_nom_opt['Island_store']

fig, ax = gm.MAA_density(tech, solutions_df,
               bins = 50, n_samples = 100000,
               plot_MAA_points = False,
                # xlim = [0, None], ylim = [0, None],
               )

# Plot optimal
ax.plot(x, y, 'o', ms = 20, color = 'tab:red')
















