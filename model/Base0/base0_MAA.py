# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 13:06:40 2023

@author: lukas
"""

import pypsa
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

import gorm as gm
import tim as tm

gm.set_plot_options()

#%% Control

Should_solve = True
Should_MAA   = True

input_name = 'base0_opt.nc'

n_snapshots   = 24*7*4
link_sum_max  = 3000

# MAA control
mga_slack     = 0.1   # 10%

# Comment out the variables that should NOT be included as MAA variables
variables = {
               # 'x1':('Generator', 'P2X'),
               # 'x2':('Generator', 'Data'),
              # 'x3':('Store',     'Store1'),
              'x4':('Link',      'link_Denmark'),
              'x5':('Link',      'link_Germany'),
              # 'x6':('Link',      'link_Belgium'),
            }
direction     = [1] * len(variables) # Create liste of ones the size of variables
mga_variables = list(variables.keys())


#%% Load and copy network

n = pypsa.Network(input_name) #Load network from netcdf file

# Reduce snapshots used for faster computing
n.snapshots = n.snapshots[:n_snapshots]
n.snapshot_weightings = n.snapshot_weightings[:n_snapshots] 

n_optimum = n.copy() # Save copy of optimum system
n_objective = n.objective

#%% Load data
# ----- Area use data ---------
area_use        = tm.get_area_use()
n.area_use      = area_use

n.link_sum_max  = link_sum_max

n.main_links    = n.links[~n.links.index.str.contains("bus")].index

#%% Custom constraints

def area_constraint(n, snapshots):
    # Get variables for all generators and store
    from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints, get_dual, get_con, write_objective, get_sol, define_variables
    
    vars_gen   = get_var(n, 'Generator', 'p_nom')
    vars_store = get_var(n, 'Store', 'e_nom')
    
    # Apply area use on variable and create linear expression 
    lhs = linexpr((n.area_use['hydrogen'], vars_gen["P2X"]), 
                  (n.area_use['data'],     vars_gen["Data"]), 
                  (n.area_use['storage'],  vars_store['Island_store']))
    
    # Define area use limit
    rhs = n.total_area #[m^2]
    
    # Define constraint
    define_constraints(n, lhs, '<=', rhs, 'Island', 'Area_Use')
    
def link_constraint(n, snapshots):
    # Create a constraint that limits the sum of link capacities
    from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints, get_dual, get_con, write_objective, get_sol, define_variables
    
    # Get link info from network
    link_names = n.main_links               # List of main link names
    link_t     = n.link_sum_max           # Maximum total link capacity
    
    # Get all link variables, and filter for only main link variables
    vars_links   = get_var(n, 'Link', 'p_nom')
    vars_links   = vars_links[link_names]
    
    # Sum up link capacities of chosen links (lhs), and set limit (rhs)
    rhs          = link_t
    lhs          = join_exprs(linexpr((1, vars_links)))
    
    #Define constraint and name it 'Total constraint'
    define_constraints(n, lhs, '=', rhs, 'Link', 'Total constraint')

#%% MAA helper functions

def assign_carriers(n):
    import pandas as pd
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
    import pandas as pd
    from pypsa.linopf import lookup, network_lopf, ilopf
    from pypsa.pf import get_switchable_as_dense as get_as_dense
    from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints, get_dual, get_con, write_objective, get_sol, define_variables
    from pypsa.descriptors import nominal_attrs
    from pypsa.descriptors import get_extendable_i, get_non_extendable_i

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
    import pandas as pd
    # from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints, get_dual, get_con, write_objective, get_sol, define_variables
    from pypsa.descriptors import nominal_attrs
    from pypsa.descriptors import get_extendable_i, get_non_extendable_i
    
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
    import numpy as np
    from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints, get_dual, get_con, write_objective, get_sol, define_variables
    
    mga_variables = options['mga_variables']
    expr_list = []
    for dir_i,var_i in zip(direction,mga_variables):
        
        if var_i[0] == 'Store':
            parameter = 'e_nom'
        else:
            parameter = 'p_nom'
        
        model_vars = get_var(n,var_i[0],parameter)[n.df(var_i[0]).carrier == var_i[1]]
            
        tmp_expr = linexpr((dir_i/len(model_vars),model_vars)).sum()
        expr_list.append(tmp_expr)

    mga_obj = join_exprs(np.array(expr_list))
    write_objective(n,mga_obj)


def get_var_values(n,mga_variables):

    variable_values = {}
    for var_i in variables:
        
        if variables[var_i][0] == 'Store':
            val = n.df(variables[var_i][0]).query('carrier == "{}"'.format(variables[var_i][1])).e_nom_opt.sum()
        else:
            val = n.df(variables[var_i][0]).query('carrier == "{}"'.format(variables[var_i][1])).p_nom_opt.sum()
        variable_values[var_i] = val

    return variable_values

def extra_functionality(n,snapshots,options,direction):
    area_constraint(n, snapshots)
    # link_constraint(n, snapshots)
    
    define_mga_constraint(n,snapshots,options['mga_slack'])
    define_mga_objective(n,snapshots,direction,options)
    
#%% MAA 
n.objective_optimum = n_objective

#%% MGA - Search 1 direction

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
            
            # n.export_to_netcdf('case_1a_res_' + str(i) + '.nc')
    
        try:
            hull = ConvexHull(solutions)
        except Exception as e:
            print(e)
    
        delta_v = hull.volume - old_volume
        old_volume = hull.volume
        epsilon = delta_v/hull.volume
        print('####### EPSILON ###############')
        print(epsilon)
else:
    pass

#%% 2D Plot

# if Should_MAA:
#     hull = ConvexHull(solutions)
    
#     plt.figure(figsize = (8,4))
    
#     DK = solutions[:,0]
#     DE = solutions[:,1]
    
#     for simplex in hull.simplices:
    
#         plt.plot(solutions[simplex, 0], solutions[simplex, 1], 'k-')
    
#     plt.plot(DK, DE,
#              'o', label = "Near-optimal")
    
#     #Plot optimal
#     plt.plot(n_optimum.generators.p_nom_opt["P2X"], 
#              n_optimum.generators.p_nom_opt["Data"],
#               '.', markersize = 20, label = "Optimal")
#     plt.xlabel("P2X capacity [MW]")
#     plt.ylabel("Data capacity [MW]")
#     plt.suptitle('MAA Analysis', fontsize = 18)
#     plt.title(f'With MGA slack = {mga_slack}', fontsize = 10)
    
#     plt.legend()

# else:
#     pass


hull = ConvexHull(solutions)

plt.figure(figsize = (8,4))

Link1 = solutions[:,0]
Link2 = solutions[:,1]

for simplex in hull.simplices:

    plt.plot(solutions[simplex, 0], solutions[simplex, 1], 'k-')

plt.plot(Link1, Link2,
          'o', label = "Near-optimal")

#Plot optimal
plt.plot(n_optimum.links.p_nom_opt["Island to Denmark"], 
          n_optimum.links.p_nom_opt["Island to Germany"],
          '.', markersize = 20, label = "Optimal")
plt.xlabel("Link1 capacity [MW]")
plt.ylabel("Link2 capacity [MW]")
plt.suptitle('MAA Analysis', fontsize = 18)
plt.title(f'With MGA slack = {mga_slack}', fontsize = 10)

#%% 3D plot

# xi = solutions[:,0]
# yi = solutions[:,1]
# zi = solutions[:,2]

# p = solutions


# #%% Alternative plot

# fig = plt.figure()

# colors = ['tab:blue', 'tab:red', 'aliceblue']
# ax = plt.axes(projection = '3d')

# ax.set_xlabel('P2X')
# ax.set_ylabel('Data')
# ax.set_zlabel('Storeage')

# # Points
# ax.plot(xi, yi, zi, 'o', c = colors[1], ms=7)

# # Define hull and edges
# hull = ConvexHull(p)
# edges = zip(*p)

# # Plot trisurface  
# from matplotlib.colors import LightSource
# ls = LightSource(azdeg=225.0, altdeg=45.0)

# ss = ax.plot_trisurf(xi, yi, zi, triangles=hull.simplices,
#                       alpha=0.8, color = colors[0],
#                       edgecolor = colors[2], linewidth = 3)
    
# fig.savefig('hull_plot_test1.svg', format = 'pdf', bbox_inches='tight')

#%%
gm.its_britney_bitch(r'C:\Users\lukas\Documents\GitHub\Masters_Thesis_NorthSeaEnergyIsland\data\Sounds')