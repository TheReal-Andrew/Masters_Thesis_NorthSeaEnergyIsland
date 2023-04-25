# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 13:06:40 2023

@author: lukas
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

# ----- Dataframe with tech data ---------
tech_df         = tm.get_tech_data(2030, 0.07)

#%% Control

Should_solve = True
Should_MAA   = True

input_name = 'v_2030_preliminary_opt.nc'

n_snapshots   = 8760 # Snapshots MUSt be the same as in the optimal solution!
link_sum_max  = 3000 # Must be the same as in the optimal solution!

# MAA control
mga_slack     = 0.1   # 10%

# Comment out the variables that should NOT be included as MAA variables
variables = {
                'x1':('Generator', 'P2X'),
                'x2':('Generator', 'Data'),
                'x3':('Store',     'Store1'),
                # 'x4':('Link',      'link_sum'),
                # 'x4':('Link',      'link_Germany'),
                # 'x6':('Link',      'link_Belgium'),
            }
direction     = [1] * len(variables) # Create liste of ones the size of variables. 1 means minimize, -1 means maximize 
mga_variables = list(variables.keys())

techs = [variables[x][1] for x in mga_variables]

#%% Load and copy network

n = pypsa.Network(input_name) #Load network from netcdf file

# Reduce snapshots used for faster computing
n.snapshots = n.snapshots[:n_snapshots]
n.snapshot_weightings = n.snapshot_weightings[:n_snapshots] 

n_optimum = n.copy() # Save copy of optimum system

n.MB        = n_optimum.generators.loc['MoneyBin'].capital_cost
n.revenue   = abs(n_optimum.generators_t.p['Data'].sum())*tech_df['marginal cost']['datacenter'] + abs(n.generators_t.p['P2X'].sum())*tech_df['marginal cost']['hydrogen']

n_objective = n.objective

#%% Load data
# ----- Area use data ---------
area_use        = tm.get_area_use()
n.area_use      = area_use

n.link_sum_max  = link_sum_max

n.main_links    = n.links[~n.links.index.str.contains("bus")].index

n.variables_set = variables


#%% MAA setup

n.objective_optimum = n_objective

def extra_functionality(n,snapshots,options,direction):
    gm.area_constraint(n, snapshots)
    gm.link_constraint(n, snapshots)
    
    gm.define_mga_constraint(n,snapshots,options['mga_slack'])
    gm.define_mga_objective(n,snapshots,direction,options)

#%% MGA - Search 1 direction
tic()
if Should_MAA:
    direction = direction # 1 means minimize, -1 means maximize 
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

    res = n.lopf(pyomo=False,
            solver_name='gurobi',
            #keep_references=True,
            #keep_shadowprices=True,
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

if Should_MAA:
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
            
            n.export_to_netcdf('forundersøgelse_MAA' + str(i) + '.nc')
    
        try:
            hull = ConvexHull(solutions)
        except Exception as e:
            print(e)
    
        delta_v = hull.volume - old_volume
        old_volume = hull.volume
        epsilon = delta_v/hull.volume
        print('####### EPSILON ###############')
        print(epsilon)

#%% Seaborn heatmap
# import numpy as np
# from numpy.linalg import det
# from scipy.stats import dirichlet
# from scipy.spatial import Delaunay

# d = gm.sample_in_hull(solutions, 10000)

# d_df = pandas.DataFrame(d,
#                         columns = techs)

# d_corr = d_df.corr()


# # Seaborn
# mask = np.triu(d_corr)
# # np.fill_diagonal(mask, False)
# import seaborn as sns

# sns.heatmap(d_corr, annot = True, linewidths = 0.5, mask = mask)

#%% 2D Subplots
print('It took ' + str(toc()) + 's to do the simulation with ' + str(len(variables)) + ' variables' )

gm.solutions_2D(techs, solutions, n_samples = 10000)

gm.solutions_heatmap2(techs, solutions)

#%% Samples dataframe and normalization
d = gm.sample_in_hull(solutions)

d_df = pandas.DataFrame(d,
                        columns = techs)

d_corr = d_df.corr()

d_corr2 = d_corr + abs(d_corr.min().min())

d_norm = d_corr2 / d_corr2.max().max()


#%% 2D Plot

if Should_MAA and solutions.shape[1] == 2:
    hull = ConvexHull(solutions)
    
    plt.figure(figsize = (8,4))
    
    x = solutions[:,0]
    y = solutions[:,1]
    
    for simplex in hull.simplices:
    
        plt.plot(solutions[simplex, 0], solutions[simplex, 1], 'k-')
    
    plt.plot(x, y,
              'o', label = "Near-optimal")
    
    plt.plot(d[:,0], d[:,1], 'o', label = 'samples')
    
    #Plot optimal
    plt.plot(n_optimum.generators.p_nom_opt["P2X"], 
              n_optimum.generators.p_nom_opt["Data"],
              '.', markersize = 20, label = "Optimal")
    plt.xlabel("P2X capacity [MW]")
    plt.ylabel("Data capacity [MW]")
    plt.suptitle('MAA Analysis', fontsize = 18)
    plt.title(f'With MGA slack = {mga_slack}', fontsize = 10)
    
    plt.legend()


#%% 3D plot

if Should_MAA and solutions.shape[1] >= 3:
    xi = solutions[:,0]
    yi = solutions[:,1]
    zi = solutions[:,2]
    
    fig = plt.figure()
    
    colors = ['tab:blue', 'tab:red', 'aliceblue']
    ax = plt.axes(projection = '3d')
    
    ax.set_xlabel(variables[list(variables)[0]][1])
    ax.set_ylabel(variables[list(variables)[1]][1])
    ax.set_zlabel(variables[list(variables)[2]][1])
    
    # Points
    ax.plot(xi, yi, zi, 'o', c = colors[1], ms=7)
    
    # Define hull and edges
    hull = ConvexHull(solutions)
    edges = zip(*solutions)
    
    # Plot trisurface  
    trisurface = ax.plot_trisurf(xi, yi, zi, triangles=hull.simplices,
                          alpha=0.8, color = colors[0],
                          edgecolor = colors[2], linewidth = 3)
    
# fig.savefig('hull_plot_test1.svg', format = 'pdf', bbox_inches='tight')

#%%
gm.its_britney_bitch(r'C:\Users\lukas\Documents\GitHub\Masters_Thesis_NorthSeaEnergyIsland\data\Sounds')