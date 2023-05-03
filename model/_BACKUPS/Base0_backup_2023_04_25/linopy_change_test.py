# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 14:17:41 2023

@author: Linda
"""

import pypsa
import linopy

#%% Load system and linopy model
n = pypsa.Network('base0_opt.nc')
n.model = linopy.read_netcdf('base0_linopy_model.nc')

#%% Change parameters

n.generators.marginal_cost['Data'] = 0

#%% Solve linopy model

# n.optimize.create_model()
# 
# n.model.constraints = model_import.constraints

# n.optimize.solve_model(solver_name = 'gurobi')