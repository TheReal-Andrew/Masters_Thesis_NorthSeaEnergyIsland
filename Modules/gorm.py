 # -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 20:21:54 2023

@author: lukas
"""
#%%
# ----- General Operation Resource Module (GORM) -----
# The GORM module contains useful helper functions for the main model.

#%% -------GENERAL FUNCTIONS -----------------------------------

# Play sound from folder
def its_britney_bitch(path = r'../../data/Sounds'):
    import os
    import random 

    files_in_path   = os.listdir(path) # Get files in directory
    file_to_play    = random.choice(files_in_path) # Choose random file to play
    
    # # Play file
    # os.startfile(path + '/' + file_to_play)
    
    # Construct full path to file
    full_path = os.path.join(path, file_to_play)
    
    # Play file
    os.startfile(full_path)
    
#%% ------- CALCULATION / OPERATION FUNCTIONS ---------------------------------

# Get annuity Annuity from discount rate (i) and lifetime (n)
def get_annuity(i, n):
    annuity = i/(1.-1./(1.+i)**n)
    return annuity

def get_annuity_snap(i, n, n_hrs = 8760):
    
    annuity = ( i/(1.-1./(1.+i)**n) ) * n_hrs/8760
    
    return annuity

# Remove outliers from dataframe, and replace with mean + n standard deviations.
def remove_outliers(df,columns,n_std):
    for col in columns:
        print('Removing outliers - Working on column: {}'.format(col))
        
        df[col][ df[col] >= df[col].mean() + (n_std * df[col].std())] = \
        df[col].mean() + (n_std * df[col].std())
        
    return df

def get_earth_distance(lat1,lat2,lon1,lon2):
    import numpy as np
    R = 6378.1  #Earths radius [km]
    dlon = (lon2 - lon1) * np.pi/180
    dlat = (lat2 - lat1) * np.pi/180
    
    lat1 = lat1 * np.pi/180
    lat2 = lat2 * np.pi/180
    
    a = (np.sin(dlat/2))**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2))**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)) 
    d = R*c #Spherical distance between two points
    
    return d

def sample_in_hull(points, n_samples = 1000):
    # From https://stackoverflow.com/questions/59073952/how-to-get-uniformly-distributed-points-in-convex-hull
    import random
    import numpy as np
    from numpy.linalg import det
    # from scipy.stats import dirichlet
    from scipy.spatial import Delaunay
    from scipy.spatial import ConvexHull
    
    dims = points.shape[-1]                     # Determine dimension of simplexes
    hull = points[ConvexHull(points).vertices]  # Find MGA points
    deln = hull[Delaunay(hull).simplices]       # Split MGA-hull into simplexes

    vols = np.abs(det(deln[:, :dims, :] - deln[:, dims:, :])) / np.math.factorial(dims) # Calculate volume of simplexes   
    
    #### Find number of samples pr. simplex from their volume
    samples_pr_simplex = [None] * vols.shape[-1]
    for k in range(vols.shape[-1]):    
        samples_pr_simplex[k] = int(np.round(vols[k]/vols.sum()*n_samples))
    
    #### Find random samples
    samples = np.zeros(shape=(sum(samples_pr_simplex),dims))
    counter = 0
    for l in range(vols.shape[-1]):
            for ll in range(samples_pr_simplex[l]):
               
                #### Find random vector which == 1
                a_list = [0,1]
                for i in range(dims): 
                    a_list.insert(i+1, random.uniform(0, 1))
                    a_list.sort()
                
                r = [None] * (dims+1)
                for j in range(len(a_list)-1):
                    r[j] = a_list[j+1] - a_list[j]
                random.shuffle(r)
                
                #### Sample the space
                sample_x = np.zeros(shape=(1,dims))

                for k in range(deln.shape[1]):
                    sample_x = np.add(deln[l][k]*r[k], sample_x)
                           
                samples[counter] = sample_x
                counter = counter + 1
        
    return samples

def get_intersection(sol1, sol2):
    import cdd as pcdd
    from scipy.spatial import ConvexHull
    import numpy as np

    v1 = np.column_stack((np.ones(sol1.shape[0]), sol1))
    mat = pcdd.Matrix(v1, number_type='fraction') # use fractions if possible
    mat.rep_type = pcdd.RepType.GENERATOR
    poly1 = pcdd.Polyhedron(mat)
    
    # make the V-representation of the second cube; you have to prepend
    # with a column of ones
    v2 = np.column_stack((np.ones(sol2.shape[0]), sol2))
    mat = pcdd.Matrix(v2, number_type='fraction')
    mat.rep_type = pcdd.RepType.GENERATOR
    poly2 = pcdd.Polyhedron(mat)

    # H-representation of the first cube
    h1 = poly1.get_inequalities()
    
    # H-representation of the second cube
    h2 = poly2.get_inequalities()

    # join the two sets of linear inequalities; this will give the intersection
    hintersection = np.vstack((h1, h2))
    
    # make the V-representation of the intersection
    mat = pcdd.Matrix(hintersection, number_type='fraction')
    mat.rep_type = pcdd.RepType.INEQUALITY
    polyintersection = pcdd.Polyhedron(mat)
    
    # get the vertices; they are given in a matrix prepended by a column of ones
    vintersection = polyintersection.get_generators()
    
    # get rid of the column of ones
    ptsintersection = np.array([
        vintersection[i][1:4] for i in range(vintersection.row_size)    
    ])
    
    # these are the vertices of the intersection; it remains to take
    # the convex hull
    intersection = ConvexHull(ptsintersection)
    
    return intersection.points

#%% ------- PyPSA FUNCTIONS -----------------------------------

# Add bidirectional link with setup for losses
def add_bi_link(network, bus0, bus1, link_name, carrier, efficiency = 1,
               capital_cost = 0, marginal_cost = 0, 
               p_nom_extendable = True, 
                p_nom_max = float('inf'), p_nom_min = 0,
               x = None, y = None, bus_shift = [0, 0]):
    # Function that adds a bidirectional link with efficiency and marginal cost
    # between two buses. This is done by adding additional "efficiency buses" 
    # (e0 and e1) for each bus, and running a bidirectional lossless link 
    # between them. The unidirectional links from e0 and e1 to their respective
    # buses then take care of adding correct marignal cost and efficiency.
    # capital cost is added on the bidirectional lossless link, while effieincy
    # and maringla cost are added on the links from efficiency buses to buses.
    
    # ---- Add efficiency buses at each end ----
    e0 = link_name + '_e0'
    e1 = link_name + '_e1'
    
    network.madd('Bus',
          names = [e0,  e1],
          x     = [network.buses.loc[network.buses.index == bus0]['x'] + bus_shift[0] , 
                   network.buses.loc[network.buses.index == bus1]['x'] + bus_shift[0]],
          y     = [network.buses.loc[network.buses.index == bus0]['y'] + bus_shift[1], 
                   network.buses.loc[network.buses.index == bus1]['y'] + bus_shift[1]],
          )
    
    # ---- main bidirectional Link ----
    # capital_cost and marginal_cost are applied here
    network.add('Link',
          link_name,
          bus0              = e0,
          bus1              = e1,
          p_min_pu          = -1,
          p_nom_extendable  = p_nom_extendable,
          capital_cost      = capital_cost,    #Capital cost is added here
          # p_nom_max         = p_nom_max,
          # p_nom_min         = p_nom_min,
          carrier           = carrier,
          )
    
    # ---- Links on bus 0 ----
    # link from buses to ebuses
    network.madd('Link',
          names         = [link_name + '_bus0_to_e0',  link_name + '_e0_to_bus0'],
          bus0          = [bus0,                       e0          ],
          bus1          = [e0,                         bus0        ],
          efficiency    = [1,                          efficiency  ],
          marginal_cost = [0,                          marginal_cost],
          p_nom_extendable  = [True, True],
          carrier       = [carrier, carrier],
          )
    
    # ---- Links on bus 0 ----
    # link from buses to ebuses
    network.madd('Link',
          names         = [link_name + '_bus1_to_e1',  link_name + '_e1_to_bus1'],
          bus0          = [bus1,                          e1          ],
          bus1          = [e1,                            bus1        ],
          efficiency    = [1,                             efficiency  ],
          marginal_cost = [0,                             marginal_cost],
          p_nom_extendable  = [True, True],
          carrier       = [carrier, carrier],
          )
    
def get_link_flow(n, connected_countries = None, plot = False):
    import pandas as pd
    # Combine powerflow from the two links that make the bidirectional link.
    # Power going from Island to DK becomes positive.
    # Power going from DK to Island becomes negative.
    
    # Set connected countries if none provided
    if connected_countries == None:
        connected_countries =  [
                                "Denmark",         
                                "Norway",          
                                "Germany",         
                                "Netherlands",     
                                "Belgium",         
                                "United Kingdom"
                                ]

    # Create a new DataFrame to store the combined rows
    result_df = pd.DataFrame()
    
    for country in connected_countries:
        p0 = n.links_t.p0[f'Island_to_{country}']
        p1 = n.links_t.p0[f'{country}_to_Island']
        
        diff = p0 - p1 # Power flow from DK to Island becomes negative
        series = pd.Series(diff,
                           name = country)
        
        result_df[country] = series
        
    if plot:
        ax = result_df.plot()
        
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(result_df.columns)/2)

    return result_df


# CONSTRAINTS

# Area constraint
def area_constraint(n, snapshots):
    # Get variables for all generators and store
    from pypsa.linopt import get_var, linexpr, define_constraints
    
    vars_gen   = get_var(n, 'Generator', 'p_nom')
    vars_store = get_var(n, 'Store', 'e_nom')
    
    # Apply area use on variable and create linear expression 
    lhs = linexpr(
                   (n.area_use['hydrogen'], vars_gen["P2X"]), 
                   (n.area_use['data'],     vars_gen["Data"]), 
                   (n.area_use['storage'],  vars_store['Storage'])
                  )
    
    # Define area use limit
    rhs = n.total_area #[m^2]
    
    # Define constraint
    define_constraints(n, lhs, '<=', rhs, 'Island', 'Area_Use')
    
def marry_links(n, snapshots):
    from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints
    
    vars_links   = get_var(n, 'Link', 'p_nom')
    
    if not hasattr(n, 'connected_countries'):
        n.connected_countries =  [
                                "Denmark",         
                                "Norway",          
                                "Germany",         
                                "Netherlands",     
                                "Belgium",         
                                "United Kingdom"
                                ]
    
    for country in n.connected_countries:
        
        lhs = linexpr((1, vars_links['Island_to_' + country]),
                      (-1, vars_links[country + '_to_Island']))
        
        define_constraints(n, lhs, '=', 0, 'Link', country + '_link_capacity_constraint')
    
def link_constraint(n, snapshots):
    # Create a constraint that limits the sum of link capacities
    from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints
    
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
    define_constraints(n, lhs, '<=', rhs, 'Link', 'Sum constraint')
    
#%% MAA FUNCTIONS

def define_mga_constraint_local(n, sns, epsilon=None, with_fix=False):
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
        rhs = n.objective_optimum + epsilon * (n.local_cost)
        
    define_constraints(n, lhs, "<=", rhs, "GlobalConstraint", "mu_epsilon")


def define_mga_constraint_links(n, sns, epsilon=None, with_fix=False):
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
        rhs = 1 * n.objective_optimum + epsilon * (n.link_total_cost + ext_const)

    define_constraints(n, lhs, "<=", rhs, "GlobalConstraint", "mu_epsilon")
    
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
    
    MB = n.MB               # Load money bin capital cost
    revenue = n.revenue     # Load the revenue from the optimal system
    
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
        rhs = (1 + epsilon) * (n.objective_optimum + revenue - MB + ext_const)

    define_constraints(n, lhs, "<=", rhs, "GlobalConstraint", "mu_epsilon")
    
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
        
def get_var_values(n,mga_variables):
    
    variables = n.variables_set

    variable_values = {}
    for var_i in variables:
        
        if variables[var_i][0] == 'Store':
            val = n.df(variables[var_i][0]).query('carrier == "{}"'.format(variables[var_i][1])).e_nom_opt.sum()
        else:
            val = n.df(variables[var_i][0]).query('carrier == "{}"'.format(variables[var_i][1])).p_nom_opt.sum()
        variable_values[var_i] = val

    return variable_values
    
#%% PLOTTING

def set_plot_options():
    import matplotlib.pyplot as plt
    import matplotlib
    color_bg      = "0.99"          #Choose background color
    color_gridaxe = "0.85"          #Choose grid and spine color
    rc = {"axes.edgecolor":color_gridaxe} 
    plt.style.use(('ggplot', rc))           #Set style with extra spines
    plt.rcParams['figure.dpi'] = 300        #Set resolution
    plt.rcParams['figure.figsize'] = [10, 5]
    matplotlib.rc('font', size=15)
    matplotlib.rc('axes', titlesize=20)
    matplotlib.rcParams['font.family'] = ['DejaVu Sans']     #Change font to Computer Modern Sans Serif
    plt.rcParams['axes.unicode_minus'] = False          #Re-enable minus signs on axes))
    plt.rcParams['axes.facecolor']= color_bg             #Set plot background color
    plt.rcParams.update({"axes.grid" : True, "grid.color": color_gridaxe}) #Set grid color
    plt.rcParams['axes.grid'] = True
    # plt.fontname = "Computer Modern Serif"
    
def get_color_codes():
    color_codes = {
                   'P2X':'#66c2a5',
                   'Data':'#fc8d62',
                   'Storage':'#8da0cb',
                   'Links':'#f3dc83',
                   'DK': '#ee6e5c',
                   'NO': '#66b8c2',
                   'DE': '#e5c494',
                   'BE': '#e5db94',
                   'NL': '#e5adcf',
                   'GB': '#b4d480',
                   }
    return color_codes
    
def plot_geomap(network, bounds = [-3, 12, 59, 50.5], size = (15,15)):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    #Plots geographical map with buses and links shown
    
    plt.rc("figure", figsize = size)   #Set plot resolution

    network.plot(
        color_geomap = True,            #Coloring on oceans
        boundaries = bounds,            #Boundaries of the plot as [x1,x2,y1,y2]
        projection=ccrs.EqualEarth()    #Choose cartopy.crs projection
        )
    
def fill_polyhedron(techs, solutions, 
                    ax = None, label = None, tech_titles = None,
                    title = None,
                    fillcolor = 'tab:blue', edgecolor = 'black'):
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize = (8,8))
        
    if tech_titles == None: 
        tech_titles = techs
        
    ax.set_xlabel(tech_titles[0], fontsize = 24)
    ax.set_ylabel(tech_titles[1], fontsize = 24)
    ax.set_title(title, color = 'black')

    # Find the convex hull of the points
    hull = ConvexHull(solutions)
    
    # Plot the points and the convex hull
    ax.plot(solutions[:,0], solutions[:,1], 'o')
    for simplex in hull.simplices:
        ax.plot(solutions[simplex, 0], solutions[simplex, 1], '-', color = edgecolor)
    
    # Fill the area inside the convex hull
    hull_poly = plt.Polygon(solutions[hull.vertices], label = label, 
                            alpha = 0.2, color = fillcolor)
    
    ax.add_patch(hull_poly)
    
    return ax
    
def MAA_density(techs, solutions,
                n_samples = 10000, bins = 25,
                linewidth = 1, density_alpha = 1,
                ax = None,
                xlim = [None, None], ylim = [None, None],
                plot_MAA_points = False, filename = None,
                tech_titles = None, show_text = True,
                color = 'black', linecolor = 'gray', 
                title = 'MAA density and polyhedron',
                ):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from scipy.spatial import ConvexHull
    import numpy as np
    import pandas as pd
    
    
    if tech_titles == None: 
        tech_titles = techs
    
    set_plot_options()
    
    solutions_df = pd.DataFrame(solutions,
                                columns = techs)
    
    tech_solutions = solutions_df[techs]
    
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize = (8,8))
    
    if show_text:
        ax.set_xlabel(tech_titles[0], fontsize = 24)
        ax.set_ylabel(tech_titles[1], fontsize = 24)
        ax.set_title(title, color = color)
    
    # MAA solutions
    x, y = tech_solutions[techs[0]],   tech_solutions[techs[1]]
    
    # Sample hull
    samples = sample_in_hull(solutions_df.values, n_samples)
    
    samples_df = pd.DataFrame(samples,
                              columns = solutions_df.columns)
    
    # Set x and y as samples for this dimension
    x_samples = samples_df[techs[0]]
    y_samples = samples_df[techs[1]]
    
    # --------  Create 2D histogram --------------------
    hist, xedges, yedges = np.histogram2d(x_samples, y_samples,
                                          bins = bins)
    
    # Create grid for pcolormesh
    x_grid, y_grid = np.meshgrid(xedges, yedges)
    
    # Create pcolormesh plot with square bins
    ax.pcolormesh(x_grid, y_grid, hist.T, cmap = 'Blues', 
                  zorder = 0, alpha = density_alpha)
    
    # Create patch to serve as hexbin label
    hb = mpatches.Patch(color = 'tab:blue')
    
    ax.grid('on')
    
    # --------  Plot hull --------------------
    hull = ConvexHull(tech_solutions.values)
    
    # plot simplexes
    for simplex in hull.simplices:
        l0, = ax.plot(tech_solutions.values[simplex, 0], tech_solutions.values[simplex, 1], 'k-', 
                color = linecolor, label = 'faces',
                linewidth = linewidth, zorder = 0)
    
    # list of legend handles and labels
    l_list, l_labels   = [l0, hb], ['Polyhedron face', 'Sample density']
    
    if plot_MAA_points:
        # Plot vertices from solutions
        l1, = ax.plot(x, y,
                  'o', label = "Near-optimal",
                  color = 'lightcoral', zorder = 2)
        l_list.append(l1)
        l_labels.append('MAA points')
        
    if show_text:
        ax.legend(l_list, l_labels, 
                  loc = 'center', ncol = len(l_list),
                  bbox_to_anchor = (0.5, -0.15), fancybox=False, shadow=False,)
    
    # Set limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    if not filename == None:
        fig.savefig(filename, format = 'pdf', bbox_inches='tight')
        
    return ax 


def scientific_formatter(x, pos):
    """Formatter function to use scientific notation for values >= 1000"""
    if x >= 1000:
        return f'{x:.0e}'
    else:
        return f'{x:.0f}'
    
def solutions_2D(techs, solutions,
                 n_samples = 1000, bins = 50, ncols = 3,
                 title = 'MAA_plot', cmap = 'Blues',
                 xlim = [None, None], ylim = [None, None],
                 xlabel = None, ylabel = None,
                 opt_system = None,
                 tech_titles = None, minmax_techs = None,
                 plot_MAA_points = False,
                 filename = None,
                 cheb = False, show_minmax = False,
                 ):
    # Take a multi-dimensional MAA polyhedron, and plot each "side" in 2D.
    # Plot the polyhedron shape, samples within and correlations.
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    import numpy as np
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D
    import seaborn as sns
    import polytope as pc
    
    pad = 5
    
    colors = get_color_codes()
    
    if tech_titles == None: 
        tech_titles = techs
        
    if minmax_techs == None: 
        minmax_techs = techs
        
    if cheb:
        p1 = pc.qhull(solutions)
        cheb_center = p1.chebXc # Chebyshev ball center
        cheb_radius = p1.chebR
    
    # Sample polyhedron
    d = sample_in_hull(solutions, n_samples)
    
    # -------- create correlation matrix --------------------------
    # Create dataframe from samples
    d_df = pd.DataFrame(d, columns = techs)
    
    # Calculate correlation and normalize
    d_corr = d_df.corr()
    
    # Calculate normalized correlation, used to color heatmap.
    d_temp = d_corr + abs(d_corr.min().min())
    d_norm = d_temp / d_temp.max().max()
    
    # -------- Set up plot ----------------------------------------
    set_plot_options()
    
    text_lift = 1.075
    
    # define the endpoints of the colormap
    red    = (1.0, 0.7, 0.6)  # light red
    yellow = (1.0, 1.0, 0.8)  # light yellow
    green  = (0.6, 1.0, 0.6)  # light green
    
    # define the colormap
    cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', [red, yellow, green])
    
    # Initialize and adjust figure
    plt.figure()
    fig, axs = plt.subplots(len(techs), len(techs), figsize = (20,15))
    fig.subplots_adjust(wspace = 0.4, hspace = 0.4)
    
    # Set top titles
    for ax, col in zip(axs[0], tech_titles):
        ax.set_title(col + '\n')
    
    # Set side titles
    for ax, row in zip(axs[:,0], tech_titles):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords = ax.yaxis.label, textcoords='offset points',
                    size = 24, ha = 'right', va = 'center',
                    rotation = 90)
    
    # -------- Plotting -------------------------------
    
    # Upper triangle of subplots
    for i in range(0, len(techs)):
        for j in range(0, i):
            
            corr = d_norm[techs[i]][techs[j]] # Is only used for coloring
            num  = d_corr[techs[i]][techs[j]] # Is shown
            
            ax = axs[j][i]
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Write correlation
            corr_text = str(round(num,2))
            ax.text(0.5, 0.5, corr_text, ha='center', va='center', fontsize=20)
            
            ax.text(0.5, text_lift, 'Correlation', ha='center', va='top',
                    transform=ax.transAxes, fontsize = 16, color = 'gray')
            
            # Change bg color according to correlation
            ax.patch.set_facecolor(cmap(corr))
    
    
    # Diagonal plotting
    for j in range(0, len(techs)):

        ax = axs[j][j]
        
        sns.histplot(d_df[techs[j]].values,
                     color = colors[techs[j]],
                     bins = bins,
                     line_kws = {'linewidth':3},
                     element = 'bars',
                     kde = True,
                     ax = ax, label = '_nolegend_',)
        
        ax.text(0.5, text_lift, 'Histogram', ha='center', va='top', 
                transform=ax.transAxes, fontsize = 16, color = 'gray')
        
        if not opt_system == None:
            ax.axvline(x = opt_system[j], 
                       color = 'gold', linestyle = '--',
                       linewidth = 4, gapcolor = 'darkorange',)
            
        if cheb:
            ax.axvline(x = cheb_center[j],
                       color = 'red', linestyle = '--',
                       linewidth = 2,)
    
    
    # lower traingle of subplots
    for j in range(0, len(techs)):
        for i in range(0, j):
            
            ax = axs[j][i]
            
            if not xlabel == None:
                ax.set_xlabel(xlabel, color = 'gray', size = 16)
                ax.set_ylabel(ylabel, color = 'gray', size = 16)
            else:
                ax.set_xlabel('Capacity [MW]', color = 'gray', size = 16)
                ax.set_ylabel('Capacity [MW]', color = 'gray', size = 16)
            
            ax.text(0.5, text_lift, 'MAA density', ha='center', va='top',
                    transform=ax.transAxes, fontsize=16, color = 'gray')
            
            # MAA solutions
            x, y = solutions[:,i],   solutions[:,j]
            
            if max(x) >= 10000:
                # Set the formatter function for the x and y axes
                ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                
            if max(y) >= 10000:
                # Set the formatter function for the x and y axes
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            
            # Set x and y as samples for this dimension
            x_samples = d[:,i]
            y_samples = d[:,j]
            
            # --------  Create 2D histogram --------------------
            hist, xedges, yedges = np.histogram2d(x_samples, y_samples,
                                                  bins = bins)
    
            # Create grid for pcolormesh
            X, Y = np.meshgrid(xedges, yedges)
            
            # Create pcolormesh plot with square bins
            ax.pcolormesh(X, Y, hist.T, cmap = 'Blues', zorder = 0)
            
            # Create patch to serve as hexbin label
            hb = mpatches.Patch(color = 'tab:blue')
            
            ax.grid('on')
            
            # --------  Plot hull --------------------
            hull = ConvexHull(solutions[:,[i,j]])
            
            # plot simplexes
            for simplex in hull.simplices:
                l0, = ax.plot(solutions[simplex, i], solutions[simplex, j], '-', 
                        color = 'silver', label = 'faces', zorder = 0)
            
            # list of legend handles and labels
            l_list, l_labels   = [l0, hb], ['Polyhedron face', 'Sample density']
            
            if plot_MAA_points:
                # Plot vertices from solutions
                l1, = ax.plot(x, y,
                          'o', label = "Near-optimal",
                          color = 'lightcoral', zorder = 2)
                l_list.append(l1)
                l_labels.append('MAA points')
                
            # Show maximum and minimum of technology capacities
            if show_minmax:
                
                minmax_df = pd.DataFrame(solutions, columns = techs)
                
                for tech in minmax_techs:
                    
                        large = minmax_df.nlargest(1, tech)
                        small = minmax_df.nsmallest(1, tech)
                        
                        ax.plot(large[techs[i]], large[techs[j]],
                                  'o',  ms = 10, zorder = 2,
                                  color = colors[tech])
                        
                        
                        ax.plot(small[techs[i]], small[techs[j]],
                                  'o', ms = 10, zorder = 2,
                                  markeredgecolor = colors[tech],
                                  markeredgewidth = 3, 
                                  markerfacecolor = 'none',)
                        
            #optimal solutions
            if not opt_system == None:
                x_opt, y_opt = opt_system[i],   opt_system[j]
                
                # Plot optimal solutions
                ax.scatter(x_opt, y_opt,
                            marker = '*', 
                            s = 1000, zorder = 4,
                            linewidth = 2, alpha = 0.85,
                            facecolor = 'gold', edgecolor = 'darkorange',)
                
                l2 = Line2D([0], [0], marker = '*', color = 'gold',
                            markeredgecolor = 'darkorange', markeredgewidth = 2,
                            markersize = 25, label = 'Optimal Solutions',
                            linestyle = '',)
                l2_2 = Line2D([0], [0], linestyle = '--', color = 'gold',
                              gapcolor = 'darkorange', linewidth = 4,)
                
                l_list.append(l2)
                l_labels.append('Optimal solution')
                l_list.append(l2_2)
                l_labels.append('Optimal line')
                
            if cheb:
                
                l3, = ax.plot(cheb_center[i], cheb_center[j],
                              marker = 'o', linestyle = '',
                              ms = 15, zorder = 3,
                              color = 'red',)
                l3_2 = Line2D([0], [0], linestyle = '--', color = 'red',
                              linewidth = 4,)
                
                l_list.append(l3)
                l_labels.append(f'Chebyshev center (r = {round(cheb_radius)})')
                l_list.append(l3_2)
                l_labels.append('Chebyshev line')
                
            # Set limits
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
    
    if show_minmax:
        for tech in minmax_techs:
            
            lm1 = Line2D([0], [0], marker = 'o', color = colors[tech],
                         ms = 10, linestyle = '',)
            
            lm2 = Line2D([0], [0], marker = 'o', 
                         markeredgecolor = colors[tech],
                         markerfacecolor = 'none',
                         ms = 10, linestyle = '',)
            
            # Replace Data with IT
            title1 = ['IT' if tech == 'Data' else tech]
            
            # Set titles
            l_list.append(lm1)
            l_labels.append(f'{title1[0]} max')
            
            l_list.append(lm2)
            l_labels.append(f'{title1[0]} min')
            
    
    # Place legend below subplots
    ax = axs[len(techs)-1, int(np.median([1,2,3]))-1] # Get center axis
    
    legend_right = 1.2 if len(techs) == 4 else 0.5
    
    ax.legend(l_list,
              l_labels, 
              loc = 'center', ncol = ncols,
              bbox_to_anchor=(legend_right, -0.15*len(techs)), fancybox=False, shadow=False,)
    
    fig.suptitle(title, fontsize = 24, y = 0.96)
    
    if not filename == None:
        fig.savefig(filename, format = 'pdf', bbox_inches='tight')
        
    return axs
        
def solutions_3D(techs, solutions,
                 figsize = (10,10),
                 markersize = 7, linewidth = 3,
                 xlim = [None, None], ylim = [None, None], zlim = [None, None],
                 filename = None):
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    
    set_plot_options()
    
    if not solutions.shape[1] == 3: # Check if solutions are 3D.
        print('Solutions are not 3-dimensional. Cannot plot. \n')
        return
    
    xi = solutions[:,0]
    yi = solutions[:,1]
    zi = solutions[:,2]
    
    fig = plt.figure(figsize = figsize)
    
    # Set colors and plot projection
    colors = ['tab:blue', 'tab:red', 'aliceblue']
    ax = plt.axes(projection = '3d')
    ax.set(xlim = xlim, ylim = ylim, zlim = zlim)
    
    # Set axis labels
    ax.set_xlabel(techs[0])
    ax.set_ylabel(techs[1])
    ax.set_zlabel(techs[2])
    
    # Define hull and edges
    hull = ConvexHull(solutions)
    
    # Plot surfaces and lines  
    ax.plot_trisurf(xi, yi, zi, 
                    triangles = hull.simplices,
                    alpha=0.8, color = colors[0],
                    edgecolor = colors[2], linewidth = linewidth)
    
    # Plot MAA points
    ax.plot(xi, yi, zi, 'o', c = colors[1], ms = markersize)
    
    if not filename == None:
        fig.savefig(filename, format = 'pdf', bbox_inches='tight')
            
def sns_heatmap(techs, solutions, triangular = False, n_samples = 1000):
    import pandas
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    plt.figure()
    
    d = sample_in_hull(solutions, n_samples)

    d_df = pandas.DataFrame(d,
                            columns = techs)

    d_corr = d_df.corr()

    mask = None
    # Seaborn
    if triangular:
        mask = np.triu(d_corr)
        np.fill_diagonal(mask, False)

    sns.heatmap(d_corr, annot = True, linewidths = 0.5, mask = mask)
    
def bake_local_area_pie(n, title = 'title', 
                        labels = None,
                        exportname = None, ax = None):
    # Create a piechart, showing the area used by each local technology on
    # the Energy Island.
    
    import matplotlib.pyplot as plt
    
    def autopct_format(values, k):
        def my_format(pct):
            
            if pct == 0:
                return ''
            
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{:.1f}%\n({v:d} m$^2$)'.format(pct, v=val)
        return my_format
    
    P2X_a   = n.generators.p_nom_opt["P2X"] * n.area_use['hydrogen']
    Data_a  = n.generators.p_nom_opt["Data"] * n.area_use['data']
    Store_a = n.stores.e_nom_opt['Storage'] * n.area_use['storage']
    
    total_A = P2X_a + Data_a + Store_a
    
    pie_data = [P2X_a, Data_a, Store_a]
    k        = [n.area_use['hydrogen'], n.area_use['data'], n.area_use['storage']] 
    labels   =  None
    
    
    if ax == None:
        fig, ax  = plt.subplots(figsize = (6,6))
        labels   =  ["P2X", "Data", "Store", "Links"]
        
    ax.pie(pie_data, 
             autopct = autopct_format(pie_data, k),
             textprops = {'fontsize': 10},
             startangle = 90)
    
    ax.axis('equal')
    ax.margins(0, 0)
    ax.text(0, 1.05, f'Area used: {total_A:.0f} m$^2$', ha='center', fontsize=10)
    
    ax.set_title(title,
              fontsize = 16,
              pad = 20)
    
    if not labels == None:
       ax.legend(labels = labels, 
                 loc = 'center', ncol = 3,
                 bbox_to_anchor=(0.5, -0.1), fancybox=False, shadow=False,)
    
    if not exportname == None and ax == None:
        fig.savefig(exportname, format = 'pdf', bbox_inches='tight')
        
def bake_capacity_pie(n, title = 'Title', exportname = None, ax = None):
    # Create a piechart, showing the capacity of all links and local demand.
    
    import matplotlib.pyplot as plt
    
    def autopct_format(values):
        def my_format(pct):
            
            if pct == 0:
                return ''
            
            total = sum(values)
            val   = int(round(pct*total/100.0))
            return '{:.1f}% \n ({v:d} MW)'.format(pct, v=val)
        return my_format
    
    P2X_capacity   = n.generators.p_nom_opt["P2X"]      # [MW]
    data_capacity  = n.generators.p_nom_opt["Data"]     # [MW]
    store_capacity = n.stores.e_nom_opt['Storage'] # [MWh]
    
    links_capacity = n.links.p_nom_opt[n.main_links].sum() # [MW]
    
    total_capacity = P2X_capacity + data_capacity + store_capacity + links_capacity
    
    pie_data = [P2X_capacity, data_capacity, store_capacity, links_capacity]
    labels   =  None
    
    if ax == None:
        fig, ax  = plt.subplots(figsize = (6,6))
        labels   =  ["P2X", "Data", "Store", "Links"]
        
    ax.pie(pie_data, 
           autopct = autopct_format(pie_data),
           textprops={'fontsize': 10},
           startangle=90)
    
    ax.axis('equal')
    ax.margins(0, 0)
    ax.text(0, 1.05, f'Total capacity: {round(total_capacity,2)} MW', ha='center', fontsize=10)
    
    ax.set_title(title,
              fontsize = 16,
              pad = 20)
    
    if not labels == None:
       ax.legend(labels = labels, 
                 loc = 'center', ncol = 4,
                 bbox_to_anchor=(0.5, -0.1), fancybox=False, shadow=False,)
    
    if not exportname == None:
        fig.savefig(exportname, format = 'pdf', bbox_inches='tight')
  

def waffles_from_values(values, title, waffletitles,
                        figsize = (10, 7), hspace = 0.35,
                        ):
    import matplotlib.pyplot as plt
    from pywaffle import Waffle
    
    fig, axs = plt.subplots(len(values), 1, figsize = figsize)
    fig.suptitle(title, fontsize = 22)
    fig.subplots_adjust(wspace = 0.5, hspace = hspace)
    
    i = 0
    for waffle_data in values:
        
        ax = axs[i]
        ax.set_aspect(aspect="equal")
        
        Waffle.make_waffle(
            ax     = ax,
            rows   = 5,
            values = waffle_data,
            title  = {"label": waffletitles[i], "loc": "left"},
            labels = [f"{k} ({v}%)" for k, v in waffle_data.items()],
            legend = {'loc': 'lower left', 'bbox_to_anchor': (0, -0.3), 'ncol': len(waffle_data), 'framealpha': 0}
            )
        i += 1
        
    return fig, axs
    
def histograms_3MAA(techs, solutions, filename = None,
                    title = 'Histograms', n_samples = 10000,
                    titlesize = 24, titley = 0.97):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    
    colors = get_color_codes()
    
    #Check if one or more solutions was passed, and create list
    if not isinstance(solutions, list):
        solutions = [solutions]
    
    fig, axs = plt.subplots(len(techs), 1, figsize = (15,4*len(techs)))
    fig.subplots_adjust(hspace = 0.5)
    fig.suptitle(title, fontsize = 32, y = titley)
    axs = axs.ravel()
    
    for solution, year in zip(solutions, [2030, 2040]):
        # Sampling
        samples = sample_in_hull(solution, n_samples)
        samples_df = pd.DataFrame(samples, columns = techs)
        
        # ----------------------- Histograms plot -----------------------------
        
        linestyle = '-' if year == 2030 else '--'
        dash_pattern = [10, 0] if year == 2030 else [10,10]
        
        for tech, ax in zip(techs, axs):
            
            # Create Seaborn histplot with KDE line
            sns.histplot(samples_df[tech].values, 
                         line_kws = {'linewidth':3, 'linestyle':linestyle,
                                     'dashes': dash_pattern},
                         element = 'step',
                         color = colors[tech],
                         alpha = 1/2,
                         kde = True,
                         ax = ax, label='_nolegend_',
                         )
                
            xUnit  = '[MWh]' if tech == 'Storage' else '[MW]'
            ax.set(xlabel = f'Installed capacity {xUnit}', 
                   ylabel = 'Frequency',
                   )
            
            axtitle = 'IT' if tech == 'Data' else tech
            
            ax.set_title(axtitle, color = colors[tech], fontsize = titlesize, y = 0.975)
        
    # ----------------------- Set legend -----------------------------
    for tech, ax in zip(techs, axs):
        
        handles = [
            Line2D([], [], color = colors[tech], linestyle='-', linewidth = 3),
            Line2D([], [], color = colors[tech], linestyle='--', linewidth = 3)
        ]
        labels = ['2030', '2040']
        ax.legend(handles=handles, labels=labels, loc='upper center',
                  ncol=2, fancybox=False, shadow=False)
        
    if not filename == None:
        fig.savefig(f'graphics/{filename}', format = 'pdf', bbox_inches='tight')
    

def solutions_2D_small(techs, solutions, chosen_techs,
                 n_samples = 1000, bins = 50, ncols = 2,
                 title = None, cmap = 'Blues',
                 xlim = [None, None], ylim = [None, None],
                 xlabel = None, ylabel = None,
                 opt_system = None, axs = None,
                 tech_titles = None, minmax_techs = None,
                 filename = None,
                 cheb = False, show_minmax = False,
                 legend_v = -0.2, legend_h = 0.5,
                 figsize = (20,5),
                 ):
    # Take a multi-dimensional MAA polyhedron, and plot each "side" in 2D.
    # Plot the polyhedron shape, samples within and correlations.
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    import numpy as np
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    import seaborn as sns
    import polytope as pc
    
    pad = 5
    
    colors = get_color_codes()
    
    tech_names = ['IT' if tech == 'Data' else tech for tech in chosen_techs]
    
    tech0 = chosen_techs[0]
    tech1 = chosen_techs[1]
    
    if title == None:
        title = f'MAA and histogram plot for MAA variables: {tech_names[0]}, {tech_names[1]}'
    
    if tech_titles == None: 
        tech_titles = techs
        
    if minmax_techs == None: 
        minmax_techs = techs
        
    if cheb:
        p1 = pc.qhull(solutions)
        cheb_center = p1.chebXc # Chebyshev ball center
        cheb_radius = p1.chebR
    
    # Sample polyhedron
    d = sample_in_hull(solutions, n_samples)
    
    # -------- create dataframes --------------------------
    # Create dataframe from samples
    samples_df = pd.DataFrame(d, columns = techs)
    
    # Create solutions dataframe
    solutions_df = pd.DataFrame(solutions, columns = techs)
    
    # -------- Set up plot ----------------------------------------
    set_plot_options()
    
    # Initialize and adjust figure
    
    if axs is None:
        plt.figure()
        
        fig, axs = plt.subplots(1, 2, figsize = figsize,
                               gridspec_kw={'width_ratios': [1, 3]},
                              )
        fig.subplots_adjust(wspace = 0.2, hspace = 0.2)
        fig.suptitle(title, fontsize = 24, y = 1.1)
        
    axs_twin = axs[1].twiny()
    
    handles, labels = [], []
    
    axs[0].set_title('MAA Density')
    axs[1].set_title('Histograms')
    
    axs_twin.spines['top'].set_position(('axes', -0.2))
    
    axs[1].spines['top'].set_position(('axes', -0.005))
    
    axs[1].set_xlabel('Capacity [MW]/[MWh]', color = 'gray',
                      labelpad = 40)
    axs[0].set(xlabel = tech0 + ' [MW]', 
               ylabel = tech1 + ' [MW]')
    
    # Sns histplots - new axis --------------------------
    
    handles, labels = [], []
    
    for ax, tech in zip([axs[1], axs_twin], chosen_techs):
        sns.histplot(samples_df[tech].values, 
                     line_kws = {'linewidth': 3},
                     element  = 'step',
                     color    = colors[tech],
                     alpha    = 1/2,
                     bins     = bins,
                     kde      = True,
                     ax       = ax,
                     label    = tech,)
        
        lpatch = mpatches.Patch(color = colors[tech],
                                alpha = 0.5)
        
        handles.append(lpatch)
        labels.append(tech)
        
        ax.spines['top'].set_edgecolor(colors[tech])
        ax.tick_params(axis = 'x', colors = colors[tech])
    
    ax.legend(handles, labels)
    
    
    # MAA Density plot - new axis --------------------------
    handles, labels = [], []
    
    # x, y = solutions_df[tech0], solutions_df[tech1]
    
    # Set x and y as samples for this dimension
    x_samples = samples_df[tech0]
    y_samples = samples_df[tech1]
    
    # --------  Create 2D histogram --------------------
    hist, xedges, yedges = np.histogram2d(x_samples, y_samples,
                                          bins = bins)

    # Create grid for pcolormesh
    X, Y = np.meshgrid(xedges, yedges)
    
    # Create pcolormesh plot with square bins
    axs[0].pcolormesh(X, Y, hist.T, cmap = 'Blues', zorder = 0)
    
    # Create patch to serve as hexbin label
    hb = mpatches.Patch(color = 'tab:blue')
    
    handles.append(hb)
    labels.append('MAA density')
    
    axs[0].grid('on')
    
    # --------  Plot hull --------------------
    hull = ConvexHull(solutions_df[[tech0, tech1]].values)
    
    # plot simplexes
    for simplex in hull.simplices:
        l0, = axs[0].plot(solutions_df[tech0][simplex],
                          solutions_df[tech1][simplex], '-', 
                color = 'silver', label = 'faces', zorder = 0)
        
    handles.append(l0)
    labels.append('Polyhedron face')
    
    # -------- Chebyshev --------------------
    if cheb:
        
        cheb_df = pd.DataFrame(np.array([cheb_center]), columns = techs)
        
        l3, = axs[0].plot(cheb_df[tech0], cheb_df[tech1],
                      marker = 'o', linestyle = '',
                      ms = 15, zorder = 3,
                      color = 'red',)
        
        handles.append(l3)
        labels.append(f'Chebyshev center (r = {round(cheb_radius)})')
    
    #optimal solutions
    if not opt_system == None:
        opt_df = pd.DataFrame(np.array([opt_system]), columns = techs)
        
        x_opt, y_opt = opt_df[tech0].values,   opt_df[tech1].values
        
        # Plot optimal solutions
        axs[0].scatter(x_opt, y_opt,
                    marker = '*', 
                    s = 1000, zorder = 4,
                    linewidth = 2, alpha = 0.85,
                    facecolor = 'gold', edgecolor = 'darkorange',)
        
        l2 = Line2D([0], [0], marker = '*', color = 'gold',
                    markeredgecolor = 'darkorange', markeredgewidth = 2,
                    markersize = 25, label = 'Optimal Solutions',
                    linestyle = '',)
        
        handles.append(l2)
        labels.append('Optimal solution')
        
    # Show maximum and minimum of technology capacities
    if show_minmax:
        
        minmax_df = pd.DataFrame(solutions, columns = techs)
        
        for tech in minmax_techs:
            
                large = minmax_df.nlargest(1, tech)
                small = minmax_df.nsmallest(1, tech)
                
                axs[0].plot(large[tech0], large[tech1],
                          'o',  ms = 10, zorder = 2,
                          color = colors[tech])
                
                
                axs[0].plot(small[tech0], small[tech1],
                          'o', ms = 10, zorder = 2,
                          markeredgecolor = colors[tech],
                          markeredgewidth = 3, 
                          markerfacecolor = 'none',)
                
                lm1 = Line2D([0], [0], marker = 'o', color = colors[tech],
                                      ms = 10, linestyle = '',)
                        
                lm2 = Line2D([0], [0], marker = 'o', 
                              markeredgecolor = colors[tech],
                              markerfacecolor = 'none',
                              ms = 10, linestyle = '',)
                
                handles.append(lm1)
                labels.append(f'{tech} max')
                
                handles.append(lm2)
                labels.append(f'{tech} min')
                
    axs[0].legend(handles, labels, loc = 'lower center',
                  ncols = ncols,
                  bbox_to_anchor=(legend_h, legend_v),)
        
    return axs
    
def MAA_density_for_vars(techs, solutions, chosen_techs,
                         n_samples = 1000, bins = 50, ncols = 2, figsize = (10,10),
                         title = None, legend_v = -0.15, legend_h = 0.5, 
                         show_legend = True, loc = 'lower center',
                         opt_system = None, ax = None,
                         tech_titles = None, minmax_techs = None,
                         filename = None, density = True, polycolor = 'silver',
                         cheb = False, show_minmax = False, minmax_legend = True,
                         ):
    # Take a multi-dimensional MAA polyhedron, and plot each "side" in 2D.
    # Plot the polyhedron shape, samples within and correlations.
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    import numpy as np
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    import seaborn as sns
    import polytope as pc
    
    colors = get_color_codes()
    
    tech_names = ['IT' if tech == 'Data' else tech for tech in chosen_techs]
    
    tech0 = chosen_techs[0]
    tech1 = chosen_techs[1]
    
    if title == None:
        title = f'MAA density plot for MAA variables: {tech_names[0]}, {tech_names[1]}'
    
    if tech_titles == None: 
        tech_titles = techs
        
    if minmax_techs == None: 
        minmax_techs = techs
        
    if cheb:
        p1 = pc.qhull(solutions)
        cheb_center = p1.chebXc # Chebyshev ball center
        cheb_radius = p1.chebR
    
    # -------- Set up plot ----------------------------------------
    set_plot_options()
    
    # Initialize and adjust figure
    
    if ax is None:
        plt.figure()
        
        fig, ax = plt.subplots(1, 1, figsize = figsize,)
        fig.subplots_adjust(wspace = 0.2, hspace = 0.2)
        fig.suptitle(title, fontsize = 24)
        
    # ax.set_title(title, fontsize = 24)
        
    ax.set(xlabel = tech0 + ' [MW]', 
           ylabel = tech1 + ' [MW]')
    
    handles, labels = [], []
    
    # ------------ Sampling ----------------------------------
    # Create solutions dataframe
    solutions_df = pd.DataFrame(solutions, columns = techs)
    
    if density:
        # Sample polyhedron
        d = sample_in_hull(solutions, n_samples)
        
        # Create dataframe from samples
        samples_df = pd.DataFrame(d, columns = techs)
        
        # Set x and y as samples for this dimension
        x_samples = samples_df[tech0]
        y_samples = samples_df[tech1]
        
        # --------  Create 2D histogram --------------------
        hist, xedges, yedges = np.histogram2d(x_samples, y_samples,
                                              bins = bins)

        # Create grid for pcolormesh
        X, Y = np.meshgrid(xedges, yedges)
        
        # Create pcolormesh plot with square bins
        ax.pcolormesh(X, Y, hist.T, cmap = 'Blues', zorder = 0)
        
        # Create patch to serve as hexbin label
        hb = mpatches.Patch(color = 'tab:blue')
        
        handles.append(hb)
        labels.append('MAA density')
        
        ax.grid('on')
    
    # --------  Plot hull --------------------
    hull = ConvexHull(solutions_df[[tech0, tech1]].values)
    
    # plot simplexes
    for simplex in hull.simplices:
        l0, = ax.plot(solutions_df[tech0][simplex],
                          solutions_df[tech1][simplex], '-', 
                color = polycolor, label = 'faces', zorder = 0)
        
    handles.append(l0)
    labels.append('Polyhedron face')
    
    # -------- Chebyshev --------------------
    if cheb:
        
        cheb_df = pd.DataFrame(np.array([cheb_center]), columns = techs)
        
        l3, = ax.plot(cheb_df[tech0], cheb_df[tech1],
                      marker = 'o', linestyle = '',
                      ms = 15, zorder = 3,
                      color = 'red',)
        
        handles.append(l3)
        labels.append(f'Chebyshev center (r = {round(cheb_radius)})')
    
    #optimal solutions
    if not opt_system == None:
        opt_df = pd.DataFrame(np.array([opt_system]), columns = techs)
        
        x_opt, y_opt = opt_df[tech0].values,   opt_df[tech1].values
        
        # Plot optimal solutions
        ax.scatter(x_opt, y_opt,
                    marker = '*', 
                    s = 1000, zorder = 4,
                    linewidth = 2, alpha = 0.85,
                    facecolor = 'gold', edgecolor = 'darkorange',)
        
        l2 = Line2D([0], [0], marker = '*', color = 'gold',
                    markeredgecolor = 'darkorange', markeredgewidth = 2,
                    markersize = 25, label = 'Optimal Solutions',
                    linestyle = '',)
        
        handles.append(l2)
        labels.append('Optimal solution')
        
    # Show maximum and minimum of technology capacities
    if show_minmax:
        
        minmax_df = pd.DataFrame(solutions, columns = techs)
        
        for tech in minmax_techs:
            
                large = minmax_df.nlargest(1, tech)
                small = minmax_df.nsmallest(1, tech)
                
                ax.plot(large[tech0], large[tech1],
                          'o',  ms = 10, zorder = 2,
                          color = colors[tech])
                
                
                ax.plot(small[tech0], small[tech1],
                          'o', ms = 10, zorder = 2,
                          markeredgecolor = colors[tech],
                          markeredgewidth = 3, 
                          markerfacecolor = 'none',)
                
                if minmax_legend:
                    lm1 = Line2D([0], [0], marker = 'o', color = colors[tech],
                                          ms = 10, linestyle = '',)
                            
                    lm2 = Line2D([0], [0], marker = 'o', 
                                  markeredgecolor = colors[tech],
                                  markerfacecolor = 'none',
                                  ms = 10, linestyle = '',)
                    
                    handles.append(lm1)
                    labels.append(f'{tech} max')
                    
                    handles.append(lm2)
                    labels.append(f'{tech} min')
                
    if show_legend:
        ax.legend(handles, labels, loc = loc,
                      ncols = ncols,
                      bbox_to_anchor=(legend_h, legend_v),)
        
    return ax

def plot_intersection(sol1, sol2, intersection = None,
                      plot_points = False, plot_edges = True,
                      colors = ['tab:blue', 'tab:red', 'aliceblue'],
                      markersize = 2, linewidth = 2):
    from scipy.spatial import ConvexHull
    import matplotlib.pyplot as plt
    
    if intersection is None:
        print('\n No intersection given, finding intersection... \n')
        
    plt.figure()
    ax = plt.axes(projection = '3d')
    
    # Plot original hulls as see-through
    for sol in [sol1, sol2]:
        
        xi = sol[:,0]
        yi = sol[:,1]
        zi = sol[:,2]
        
        # Define hull and edges
        hull = ConvexHull(sol)
        
        # Plot surfaces and lines  
        ax.plot_trisurf(xi, yi, zi, 
                        triangles = hull.simplices,
                        alpha=0.1, color = colors[0],
                        linewidth = 0)
    
    
    if intersection is None:
        intersection = get_intersection(sol1, sol2)
        print('\n Intersection found \n')
    
    # Plot intersection
    sol = intersection
    
    xi = sol[:,0]
    yi = sol[:,1]
    zi = sol[:,2]
    
    # Define hull and edges
    hull = ConvexHull(sol)
    
    linewidth = linewidth if plot_edges else 0
    
    # Plot surfaces and lines  
    ax.plot_trisurf(xi, yi, zi, 
                    triangles = hull.simplices,
                    alpha=0.8, color = colors[0],
                    edgecolor = colors[2], linewidth = linewidth)
    
    if plot_points:
        # Plot intersection points
        ax.plot(xi, yi, zi, 'o', c = colors[1], ms = markersize,)
        
def waffles_area_and_capacity(n, 
                              title = 'Area and Capacity waffle diagrams',
                              filename = None,):
    import matplotlib.pyplot as plt
    from pywaffle import Waffle
    
    # ----- Initial data loading and processing ----
    # Get colors
    col = get_color_codes()
    col_a = [col['P2X'], col['Data'], col['Storage']]
    col_c = [col['P2X'], col['Data'], col['Storage'], col['Links']]
    
    # Area data
    P2X_a   = n.generators.p_nom_opt["P2X"] * n.area_use['hydrogen']
    Data_a  = n.generators.p_nom_opt["Data"] * n.area_use['data']
    Store_a = n.stores.e_nom_opt['Storage'] * n.area_use['storage']
    
    total_a = P2X_a + Data_a + Store_a
    
    val_a  = {'P2X':    (P2X_a/total_a * 100), 
              'IT':     (Data_a/total_a * 100), 
              'Storage':(Store_a/total_a * 100)}
    
    # Capacity data
    P2X_c   = n.generators.p_nom_opt["P2X"] 
    Data_c  = n.generators.p_nom_opt["Data"] 
    Store_c = n.stores.e_nom_opt['Storage'] 
    Links_c = n.links.p_nom_opt[n.main_links].sum()
    
    total_c = P2X_c + Data_c + Store_c + Links_c
    
    val_c  = {'P2X':    (P2X_c/total_c * 100), 
              'IT':     (Data_c/total_c * 100), 
              'Storage':(Store_c/total_c * 100),
              'Links':  (Links_c/total_c * 100)}
    
    # ----- Create waffle diagrams ---------------------------
    # Create waffle diagram
    fig = plt.figure(
        FigureClass = Waffle,
        plots = {
                311: {
                     'values': val_a,
                     'title':  {'label': f'Area use distribution - Area available: {round(total_a)} $m^2$', 'loc': 'left'},
                     'labels': [f"{k} ({int(v)}%) \n {round(v/100*total_a)} $m^2$" for k, v in val_a.items()],
                     'legend': {'loc': 'lower left', 'bbox_to_anchor': (0, -0.325), 'ncol': len(val_a), 'framealpha': 0},
                     'colors': col_a,
                      },
                312: {
                     'values': val_c,
                     'title':  {'label': f'Installed capacity distribution - Total capacity: {round(total_c/1000, 1)} $GW$', 'loc': 'left'},
                     'labels': [f"{k} ({int(v)}%) \n {round(v/100*total_c/1000, 1)} $GW$" for k, v in val_c.items()],
                     'legend': {'loc': 'lower left', 'bbox_to_anchor': (0, -0.325), 'ncol': len(val_c), 'framealpha': 0},
                     'colors': col_c,
                      },
            },
        rows   = 5,
        columns = 20,
        figsize = (10,12),
    )
    
    # Set title
    fig.suptitle(title, 
                  fontsize = 28)
    
    # Save if filename is given
    if filename is not None:
        fig.savefig('graphics/'+filename, format = 'pdf', bbox_inches='tight')
        
    return fig