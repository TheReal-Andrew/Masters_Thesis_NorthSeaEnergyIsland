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
study_name = 'sensitivity_sweep'

# Comment out the variables that should NOT be included as MAA variables
variables = {
                'x1':('Generator', 'P2X'),
                'x2':('Generator', 'Data'),
                'x3':('Store',     'Storage'),
                # 'x4':('Link',      'link_Denmark'),
                # 'x5':('Link',      'link_Norway'),
                # 'x6':('Link',      'link_Germany'),
                # 'x7':('Link',      'link_Netherlands'),
                # 'x8':('Link',      'link_Belgium'),
                # 'x9':('Link',      'link_United Kingdom'),
            }

input_name        = f'v_{year}_{study_name}_opt.nc'
MAA_network_names = f'v_{year}_{study_name}_{len(variables)}MAA_{int(mga_slack*100)}p_'
MAA_solutions     = f'v_{year}_{study_name}_{len(variables)}MAA_{int(mga_slack*100)}p_'

#%% Load and copy network

n = pypsa.Network(input_name) #Load network from netcdf file

n_optimum   = n.copy() # Save copy of optimum system
n_objective = n.objective # Save optimum objective

#%% Load data to recreate constraints
# ----- Dataframe with tech data ---------
tech_df         = tm.get_tech_data(year, 0.07)

# ---- Main parameters loaded from tim module --------
mp, mp_gen = tm.get_main_parameters()
island_area  = mp[year]['island_area'] # [m^2] total island area

# List of connected countries in the optimum solution
connected_countries =  [
                        "Denmark",         
                        "Norway",          
                        "Germany",         
                        "Netherlands",     
                        "Belgium",         
                        "United Kingdom"
                        ]

# ----- Save data in network ---------
n.area_use            = tm.get_area_use()
n.total_area          = island_area 
n.link_sum_max        = n.generators.p_nom_max['Wind']
n.main_links          = n.links.loc[n.links.bus0 == "Energy Island"].index
n.variables_set       = variables
n.connected_countries = connected_countries

# Save variables for objective function modification
n.MB        = n_optimum.generators.loc['MoneyBin'].capital_cost
n.revenue   = abs(n_optimum.generators_t.p['Data'].sum())*tech_df['marginal cost']['datacenter'] + abs(n.generators_t.p['P2X'].sum())*tech_df['marginal cost']['hydrogen']

# Calculate total local demand costs, NOT revenues
mga_variables = list(variables.keys())
techs = [variables[x][1] for x in mga_variables]

local_cost = 0
for tech in techs:
    
    if tech == 'Storage' and not n.stores.e_nom_opt[tech] == 0:
        
        capital_cost  = n.stores.capital_cost[tech]
        e_nom         = n.stores.e_nom_opt[tech]
        marginal_cost = n.stores.marginal_cost[tech]
        e_sum         = n.stores_t.e[tech].sum()
        
        local_cost += capital_cost * e_nom + marginal_cost * e_sum

    elif tech == 'Data' or tech == 'P2X':
        capital_cost = n.generators.capital_cost[tech]
        p_nom        = n.generators.p_nom_opt[tech]
        # marginal cost not included since that is revenue
        
        local_cost += capital_cost * p_nom
        
n.local_cost = local_cost

n_backup = n.copy()

#%% MAA - find directions and search

def search_direction(direction,mga_variables):
    
    options = dict(mga_slack = mga_slack,
                    mga_variables = [variables[v] for v in mga_variables])

    n.lopf(pyomo          = False,
           solver_name    = 'gurobi',
           skip_objective = True,
           solver_options = {'LogToConsole':0,
                             'crossover':0,
                             'BarConvTol' : 1e-6,                 
                             'FeasibilityTol' : 1e-2,
                             'threads':2},
            extra_functionality = lambda n,s: extra_functionality(n, s, options, direction)
        )

    all_variable_values = gm.get_var_values(n, mga_variables)
    print(all_variable_values)

    return [all_variable_values[v] for v in mga_variables]

tic()

percent_range = np.arange(0.5, 1.6, 0.1)

for percent in percent_range:
    
    for tech in techs:
        
        if tech == 'Data' and percent > 1:
            continue
        
        n = n_backup.copy()
        
        if tech == 'Storage':
            n.stores.capital_cost[tech] = n.stores.capital_cost[tech] * percent
        else:
            n.generators.capital_cost[tech] = n.generators.capital_cost[tech] * percent
        
        # ----- Save data in network ---------
        n.area_use            = tm.get_area_use()
        n.total_area          = island_area 
        n.link_sum_max        = n.generators.p_nom_max['Wind']
        n.main_links          = n.links.loc[n.links.bus0 == "Energy Island"].index
        n.variables_set       = variables
        n.connected_countries = connected_countries
        
        n.local_cost = local_cost

        #%% MAA setup

        direction     = [1] * len(variables) # Create liste of ones the size of variables. 1 means minimize, -1 means maximize 

        n.objective_optimum = n_objective

        def extra_functionality(n,snapshots,options,direction):
            gm.area_constraint(n, snapshots)
            gm.link_constraint(n, snapshots)
            
            gm.marry_links(n, snapshots)
            
            gm.define_mga_constraint_local(n,snapshots,options['mga_slack'])
            gm.define_mga_objective(n,snapshots,direction,options)

        #%% MGA - Search 1 direction

        direction = direction         # 1 means minimize, -1 means maximize 
        mga_variables = mga_variables # The variables that we are investigating

        options = dict(mga_slack=mga_slack,
                        mga_variables=[variables[v] for v in mga_variables])

        res = n.lopf(pyomo        = False,
                solver_name       = 'gurobi',
                keep_references   = True,
                keep_shadowprices = True,
                skip_objective    = True,
                solver_options    = {'LogToConsole':0,
                                     'crossover':0,
                                     'BarConvTol' : 1e-6,                 
                                     'FeasibilityTol' : 1e-2,
                                     'threads':2},
                extra_functionality = lambda n,s: extra_functionality(n,s,options,direction)
            )

        all_variable_values = gm.get_var_values(n,mga_variables)
        print(all_variable_values)

        MAA_convergence_tol = 0.05 # How much the volume stops changing before we stop, in %
        dim                 = len(mga_variables) # number of dimensions 
        dim_fullD           = len(variables)
        old_volume          = 0 
        epsilon             = 1
        directions_searched = np.empty([0,dim])
        hull                = None
        computations        = 0
        
        solutions           = np.empty(shape=[0,dim])
        
        j = 0
        while epsilon>MAA_convergence_tol:
        
            if len(solutions) <= 1: # If initial loop, use MGA directions
                directions = np.concatenate([np.diag(np.ones(dim)),-np.diag(np.ones(dim))],axis=0)
            else : # Else use hull to get new directions.
                directions = -np.array(hull.equations)[:,0:-1]
                
            directions_searched = np.concatenate([directions_searched,directions],axis=0)
        
            # Run computations in series
            i = 0
            
            for direction_i in directions:
                i        += 1
                res       = search_direction(direction_i, mga_variables)
                solutions = np.append(solutions, np.array([res]), axis=0)
                
                # n.export_to_netcdf('results/' + MAA_network_names + str(j)+ '-'+ str(i) + '.nc')
                print(f'\n #### Exported MAA network: Loop {j}, direction {i} ####    \
                        \n Directions in this loop: {len(directions)}   \
                        \n Current Epsilon: {epsilon} \n')
        
            try:
                hull = ConvexHull(solutions)
            except Exception as e:
                print(e)
        
            j += 1
        
            delta_v    = hull.volume - old_volume
            old_volume = hull.volume
            epsilon    = delta_v/hull.volume
            print('\n ####### EPSILON ###############')
            print(' ' + str(epsilon) + '\n')
            
        np.save( 'sweep_results/' + MAA_solutions + f'sweep_{tech}_{percent}_' + 'solutions.npy', solutions)
        
print(f'\n MAA sweep took {round(toc(), 2)} sec for {len(variables)} variables. \n')
