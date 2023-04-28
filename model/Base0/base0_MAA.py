# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 13:06:40 2023

@author: lukas
"""

import os
import sys
# Add modules folder to path
os.chdir(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.abspath('../../modules')) 

import pypsa
import numpy as np
from scipy.spatial import ConvexHull

import gorm as gm
import tim as tm
from ttictoc import tic,toc

gm.set_plot_options()


#%% Control

year       = 2030
mga_slack  = 0.1   # MAA slack control
study_name = 'base0'

# Comment out the variables that should NOT be included as MAA variables
variables = {
                'x1':('Generator', 'P2X'),
                'x2':('Generator', 'Data'),
                # 'x3':('Store',     'Store1'),
                # 'x4':('Link',      'link_sum'),
                # 'x4':('Link',      'link_Germany'),
                # 'x6':('Link',      'link_Belgium'),
            }

input_name        = f'v_{year}_{study_name}_opt.nc'
MAA_network_names = f'v_{year}_{study_name}_{len(variables)}MAA_{int(mga_slack*100)}p_'
MAA_solutions     = f'v_{year}_{study_name}_{len(variables)}MAA_{int(mga_slack*100)}p_'

#%% Load and copy network

n = pypsa.Network(input_name) #Load network from netcdf file

n_optimum   = n.copy() # Save copy of optimum system
n_objective = n.objective # Save optimum objective

#%% Load data
# ----- Dataframe with tech data ---------
tech_df         = tm.get_tech_data(year, 0.07)

# ----- Save data in network ---------
n.area_use      = tm.get_area_use()
n.link_sum_max  = n.generators.p_nom_max['Wind']
n.main_links    = n.links[~n.links.index.str.contains("bus")].index
n.variables_set = variables

# Save variables for objective function modification
n.MB        = n_optimum.generators.loc['MoneyBin'].capital_cost
n.revenue   = abs(n_optimum.generators_t.p['Data'].sum())*tech_df['marginal cost']['datacenter'] + abs(n.generators_t.p['P2X'].sum())*tech_df['marginal cost']['hydrogen']


#%% MAA setup

direction     = [1] * len(variables) # Create liste of ones the size of variables. 1 means minimize, -1 means maximize 
mga_variables = list(variables.keys())

techs = [variables[x][1] for x in mga_variables]

n.objective_optimum = n_objective

def extra_functionality(n,snapshots,options,direction):
    gm.area_constraint(n, snapshots)
    gm.link_constraint(n, snapshots)
    
    gm.define_mga_constraint(n,snapshots,options['mga_slack'])
    gm.define_mga_objective(n,snapshots,direction,options)

#%% MGA - Search 1 direction
tic()


direction = direction         # 1 means minimize, -1 means maximize 
mga_variables = mga_variables # The variables that we are investigating

options = dict(mga_slack=mga_slack,
                mga_variables=[variables[v] for v in mga_variables])

res = n.lopf(pyomo=False,
        solver_name='gurobi',
        keep_references=True,
        keep_shadowprices=True,
        skip_objective=True,
        solver_options={'LogToConsole':0,
                'crossover':0,
                'BarConvTol' : 1e-6,                 
                'FeasibilityTol' : 1e-2,
            'threads':2},
        extra_functionality=lambda n,s: extra_functionality(n,s,options,direction)
    )

all_variable_values = gm.get_var_values(n,mga_variables)
print(all_variable_values)

#%% MAA - find directions and search

def search_direction(direction,mga_variables):
    
    options = dict(mga_slack = mga_slack,
                    mga_variables = [variables[v] for v in mga_variables])

    n.lopf(pyomo=False,
            solver_name='gurobi',
            skip_objective=True,
            solver_options={'LogToConsole':0,
                    'crossover':0,
                    'BarConvTol' : 1e-6,                 
                    'FeasibilityTol' : 1e-2,
                'threads':2},
            extra_functionality=lambda n,s: extra_functionality(n, s, options, direction)
        )

    all_variable_values = gm.get_var_values(n, mga_variables)

    return [all_variable_values[v] for v in mga_variables]


MAA_convergence_tol = 0.05 # How much the volume stops changing before we stop, in %
dim=len(mga_variables) # number of dimensions 
dim_fullD = len(variables)
old_volume = 0 
epsilon = 1
directions_searched = np.empty([0,dim])
hull = None
computations = 0

solutions = np.empty(shape=[0,dim])

while epsilon>MAA_convergence_tol:

    if len(solutions) <= 1:
        directions = np.concatenate([np.diag(np.ones(dim)),-np.diag(np.ones(dim))],axis=0)
    else :
        directions = -np.array(hull.equations)[:,0:-1]
    directions_searched = np.concatenate([directions_searched,directions],axis=0)

    # Run computations in series
    i = 0
    
    for direction_i in directions:
        i = i + 1
        res = search_direction(direction_i,mga_variables)
        solutions = np.append(solutions,np.array([res]),axis=0)
        
        n.export_to_netcdf(MAA_network_names + str(i) + '.nc')

    try:
        hull = ConvexHull(solutions)
    except Exception as e:
        print(e)

    delta_v = hull.volume - old_volume
    old_volume = hull.volume
    epsilon = delta_v/hull.volume
    print('####### EPSILON ###############')
    print(epsilon)
    
np.save(MAA_solutions + 'solutions.npy', solutions)


print('It took ' + str(toc()) + 's to do the simulation with ' + str(len(variables)) + ' variables' )

#%%
gm.its_britney_bitch()