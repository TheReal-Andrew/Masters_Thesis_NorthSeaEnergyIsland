# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 12:41:30 2022

@author: lukas
"""

Should_MGA   = True
Should_MAA   = True
n_snapshots  = 8760
mga_slack    = 0.1

import time
startTime = time.time()

#%% Import
import pypsa 
import numpy as np

import os
import sys
# Add modules folder to path
os.chdir(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.abspath('../../modules')) 

from pypsa.descriptors import nominal_attrs
from pypsa.pf import get_switchable_as_dense as get_as_dense
from pypsa.descriptors import get_extendable_i, get_non_extendable_i
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import ConvexHull

from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints, write_objective
from pypsa.linopf import lookup

import gorm as gm
import tim as tm

#%% Load and solve network

n = pypsa.Network('v_2030_links_test_opt.nc') #Load network from netcdf file
# n = pypsa.Network('case2_direct.nc') #Load network from netcdf file
n_objective = n.objective

# Reduce snapshots used for faster computing
n.snapshots = n.snapshots[:n_snapshots]
n.snapshot_weightings = n.snapshot_weightings[:n_snapshots] 

# Change carrier for optimization
n.links.loc[["Island to Denmark","Island to Belgium"], "carrier"] = ["DC1", "DC2"]
# n.links.loc[["Island_to_Denmark","Island_to_Belgium"], "carrier"] = ["DC1", "DC2"]

n_optimum = n.copy()


#%% MGA - Variables

n.objective_optimum = n_objective

variables = {'x1':('Link','DC1'),
             'x2':('Link','DC2'),
            }

#%% Load data
# # ----- Dataframe with tech data ---------
tech_df         = tm.get_tech_data(2030, 0.07)

# # ----- Save data in network ---------
n.area_use      = tm.get_area_use()
n.link_sum_max  = n.generators.p_nom_max['Wind']
n.main_links    = n.links[~n.links.index.str.contains("bus")].index
n.variables_set = variables

# Save variables for objective function modification
n.MB        = 0 #n_optimum.generators.loc['MoneyBin'].capital_cost
n.revenue   = 0 #abs(n_optimum.generators_t.p['Data'].sum())*tech_df['marginal cost']['datacenter'] + abs(n.generators_t.p['P2X'].sum())*tech_df['marginal cost']['hydrogen']

#%% MGA functions
def extra_functionality(n,snapshots,options,direction):
    # gm.area_constraint(n, snapshots)
    gm.link_constraint(n, snapshots)
    
    define_mga_constraint(n,snapshots,options['mga_slack'])
    define_mga_objective(n,snapshots,direction,options)

def assign_carriers(n):
    """
    Author: Fabian Neumann 
    Source: https://github.com/PyPSA/pypsa-eur-mga
    """

    if "Load" in n.carriers.index:
        n.carriers = n.carriers.drop("Load")

    if "carrier" not in n.lines:
        n.lines["carrier"] = "AC"

    if n.links.empty:
        n.links["carrier"] = pd.Series(dtype=str)

    config = {
        "AC": {"color": "rosybrown", "nice_name": "HVAC Line"},
        "DC": {"color": "darkseagreen", "nice_name": "HVDC Link"},
    }
    for c in ["AC", "DC"]:
        if c in n.carriers.index:
            continue
        n.carriers = n.carriers.append(pd.Series(config[c], name=c))

def define_mga_constraint(n, sns, epsilon=None, with_fix=False):
    """
    Author: Fabian Neumann 
    Source: https://github.com/PyPSA/pypsa-eur-mga
    
    Build constraint defining near-optimal feasible space
    Parameters
    ----------
    n : pypsa.Network
    sns : Series|list-like
        snapshots
    epsilon : float, optional
        Allowed added cost compared to least-cost solution, by default None
    with_fix : bool, optional
        Calculation of allowed cost penalty should include cost of non-extendable components, by default None
    """

    if epsilon is None:
        epsilon = float(snakemake.wildcards.epsilon)

    if with_fix is None:
        with_fix = snakemake.config.get("include_non_extendable", True)

    expr = []

    # operation
    for c, attr in lookup.query("marginal_cost").index:
        cost = (
            get_as_dense(n, c, "marginal_cost", sns)
            .loc[:, lambda ds: (ds != 0).all()]
            .mul(n.snapshot_weightings.loc[sns,'objective'], axis=0)
        )
        if cost.empty:
            continue
        expr.append(linexpr((cost, get_var(n, c, attr).loc[sns, cost.columns])).stack())

    # investment
    for c, attr in nominal_attrs.items():
        cost = n.df(c)["capital_cost"][get_extendable_i(n, c)]
        if cost.empty:
            continue
        expr.append(linexpr((cost, get_var(n, c, attr)[cost.index])))

    lhs = pd.concat(expr).sum()

    if with_fix:
        ext_const = objective_constant(n, ext=True, nonext=False)
        nonext_const = objective_constant(n, ext=False, nonext=True)
        rhs = (1 + epsilon) * (n.objective_optimum + ext_const + nonext_const) - nonext_const
    else:
        ext_const = objective_constant(n)
        rhs = (1 + epsilon) * (n.objective_optimum + ext_const)

    define_constraints(n, lhs, "<=", rhs, "GlobalConstraint", "mu_epsilon")

def objective_constant(n, ext=True, nonext=True):
    """
    Author: Fabian Neumann 
    Source: https://github.com/PyPSA/pypsa-eur-mga
    """

    if not (ext or nonext):
        return 0.0

    constant = 0.0
    for c, attr in nominal_attrs.items():
        i = pd.Index([])
        if ext:
            i = i.append(get_extendable_i(n, c))
        if nonext:
            i = i.append(get_non_extendable_i(n, c))
        constant += n.df(c)[attr][i] @ n.df(c).capital_cost[i]

    return constant


def define_mga_objective(n,snapshots,direction,options):
    mga_variables = options['mga_variables']
    expr_list = []
    for dir_i,var_i in zip(direction,mga_variables):
        model_vars = get_var(n,var_i[0],'p_nom')[n.df(var_i[0]).carrier == var_i[1]]
        tmp_expr = linexpr((dir_i/len(model_vars),model_vars)).sum()
        expr_list.append(tmp_expr)

    mga_obj = join_exprs(np.array(expr_list))
    write_objective(n,mga_obj)


def get_var_values(n,mga_variables):

    variable_values = {}
    for var_i in variables:
        val = n.df(variables[var_i][0]).query('carrier == "{}"'.format(variables[var_i][1])).p_nom_opt.sum()
        variable_values[var_i] = val

    return variable_values


#%% MGA - Search 1 direction

if Should_MGA or Should_MAA:
    direction = [1,1] # 1 means minimize, -1 means maximize 
    mga_variables = ['x1','x2'] # The variables that we are investigating
    
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
    
    all_variable_values = get_var_values(n,mga_variables)
    print(all_variable_values)
else:
    pass

#%% MAA Functions
def search_direction(direction,mga_variables):
    options = dict(mga_slack=mga_slack,
                    mga_variables=[variables[v] for v in mga_variables])

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
            extra_functionality=lambda n,s: extra_functionality(n,s,options,direction)
        )

    all_variable_values = get_var_values(n,mga_variables)

    return [all_variable_values[v] for v in mga_variables]

#%% MAA - Run

if Should_MAA:
    MAA_convergence_tol = 0.01 # How much the volume stops changing before we stop, in %
    dim=len(mga_variables) # number of dimensions 
    dim_fullD = len(variables)
    old_volume = 0 
    epsilon = 1
    directions_searched = np.empty([0,dim])
    hull = None
    computations = 0
    
    solutions = np.empty(shape=[0,dim])
    
    j = 0
    while epsilon>MAA_convergence_tol:
    
        if len(solutions) <= 1:
            directions = np.concatenate([np.diag(np.ones(dim)),-np.diag(np.ones(dim))],axis=0)
            print(f'\n ### Directions in initial MGA search: {len(directions)} \n')
        else :
            directions = -np.array(hull.equations)[:,0:-1]
        directions_searched = np.concatenate([directions_searched,directions],axis=0)
    
        # Run computations in series
        i = 0
        
        for direction_i in directions:
            i += 1
            res = search_direction(direction_i,mga_variables)
            solutions = np.append(solutions,np.array([res]),axis=0)
            
            n.export_to_netcdf('case_2di_res_' + str(i) + '.nc')
            print(f'\n #### MAA iteration finished \n Searching direction {i+1} in loop {j} \n Directions in this loop: {len(directions)} \n')
    
        try:
            hull = ConvexHull(solutions)
        except Exception as e:
            print(e)

        j += 1
    
        delta_v = hull.volume - old_volume
        old_volume = hull.volume
        epsilon = delta_v/hull.volume
        print('####### EPSILON ###############')
        print(epsilon)
else:
    pass

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))

#%% Plot Hull

# if Should_MAA:
#     Hull = ConvexHull(solutions)
    
#     plt.figure()
    
#     DK = solutions[:,0]
#     DE = solutions[:,1]
    
#     plt.plot(DK/1000, DE/1000,
#              'o', label = "near-optimal",
#              markersize=10)
    
#     #Plot optimal
#     plt.plot(n_optimum.links.p_nom_opt["Island_to_Denmark"]/1000, 
#              n_optimum.links.p_nom_opt["Island_to_Belgium"]/1000,
#              '.', markersize = 30, label = "optimum")
    
#     plt.xticks(fontsize=25)
#     plt.yticks(fontsize=25)
    
#     plt.xlabel(n.links.index[0]+' capacity [GW]', fontsize=35)
#     plt.ylabel(n.links.index[1]+' capacity [GW]', fontsize=35)
    
#     plt.xlim(-1,80)
#     plt.ylim(-1,80)
    
#     plt.legend(fontsize=35)
#     plt.grid() 
    
    
#     plt.suptitle('MAA for Bi-direct links', fontsize = 40)
#     plt.title('MGA slack = ' + str(mga_slack) + ', execution time = ' + str(round(executionTime)) + ' s', fontsize = 35)
    
#     for simplex in hull.simplices:
    
#         plt.plot(solutions[simplex, 0]/1000, solutions[simplex, 1]/1000, 'k-')
    
#     plt.tight_layout()
#     plt.savefig('MGA_slack_=_' + str(mga_slack) + '_Bi-direct_links.eps')
# else:
#     pass

np.save('case_2bi_MAA_solutions_10pct', solutions)


