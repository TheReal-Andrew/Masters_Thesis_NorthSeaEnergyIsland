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
def its_britney_bitch(path):
    import os
    import random 

    files_in_path   = os.listdir(path) # Get files in directory
    file_to_play    = random.choice(files_in_path) # Choose random file to play
    
    # Play file
    os.startfile(path + '/' + file_to_play)
    
#%% ------- CALCULATION / OPERATION FUNCTIONS ---------------------------------

# Get annuity Annuity from discount rate (i) and lifetime (n)
def get_annuity(i, n):
    annuity = i/(1.-1./(1.+i)**n)
    return annuity

# Remove outliers from dataframe, and replace with n standard deviations.
def remove_outliers(df,columns,n_std):
    for col in columns:
        print('Removing outliers - Working on column: {}'.format(col))
        
        df[col][ df[col] >= df[col].mean() + (n_std * df[col].std())] = \
        df[col].mean() + (n_std * df[col].std())
        
    return df

def get_earth_distance(lat1,lat2,lon1,lon2):
    import numpy as np
    R = 6378.1  #Earths radius
    dlon = (lon2 - lon1) * np.pi/180
    dlat = (lat2 - lat1) * np.pi/180
    
    lat1 = lat1 * np.pi/180
    lat2 = lat2 * np.pi/180
    
    a = (np.sin(dlat/2))**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2))**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)) 
    d = R*c #Spherical distance between two points
    
    return d

def sample_in_hull(points, n = 1000):
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
    sample_pr_simplex = [None] * vols.shape[-1]
    for k in range(vols.shape[-1]):    
        sample_pr_simplex[k] = int(np.round(vols[k]/vols.sum()*1000))
    
    #### Find random samples
    samples = np.zeros(shape=(n,dims))
    counter = 0
    for l in range(vols.shape[-1]):
            for ll in range(sample_pr_simplex[l]):
               
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
    
#%% ------- PyPSA FUNCTIONS -----------------------------------

# Add bidirectional link with setup for losses
def add_bi_link(network, bus0, bus1, link_name, carrier, efficiency = 1,
               capital_cost = 0, marginal_cost = 0, 
               p_nom_extendable = True, p_nom_max = float('inf'),
               p_nom_min = 0,
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
          p_nom_max         = p_nom_max,
          p_nom_min         = p_nom_min,
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
          )
    
# CONSTRAINTS

# Area constraint
def area_constraint(n, snapshots):
    # Get variables for all generators and store
    from pypsa.linopt import get_var, linexpr, define_constraints
    
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
    define_constraints(n, lhs, '=', rhs, 'Link', 'Sum constraint')
    
#%% MAA FUNCTIONS
    
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
    
def solutions_2D(techs, solutions, n_samples = 1000):
    import pandas
    import matplotlib
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    
    d = sample_in_hull(solutions, n_samples)
    
    d_df = pandas.DataFrame(d,
                            columns = techs)

    d_corr = d_df.corr()
    d_corr2 = d_corr + abs(d_corr.min().min())
    d_norm = d_corr2 / d_corr2.max().max()

    
    set_plot_options()
    cmap = matplotlib.cm.get_cmap('Spectral')
    
    n_vars = len(techs)
    
    plt.figure()
    fig, axs = plt.subplots(n_vars, n_vars, figsize = (20,10))
    fig.subplots_adjust(wspace = 0.4, hspace = 0.4)
    
    for ax, col in zip(axs[0], techs):
        ax.set_title(col + '\n')
        
        
    pad = 5
    
    for ax, row in zip(axs[:,0], techs):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
    
    # Create hull
    hull = ConvexHull(solutions)
    
    for i in range(0,len(techs)):
        
        for j in range(0,len(techs)):
            
            ax = axs[j][i]
            
            corr = d_norm[techs[i]][techs[j]]
            num  = d_corr[techs[i]][techs[j]]
            
            if i == j:
                # ax.axis('off')
                ax.text(1, 1, str(round(num,2)), ha='left', va='bottom', 
                        bbox={'facecolor': cmap(corr), 'alpha': 1, 'pad': 5})
                
                continue
            
            x = solutions[:,i]
            y = solutions[:,j]
            
            for simplex in hull.simplices:
            
                ax.plot(solutions[simplex, i], solutions[simplex, j], 'k-', zorder = 1)
                
            ax.plot(x, y,
                      'o', label = "Near-optimal", zorder = 2)
            
            ax.plot(d[:,i], d[:,j], 'o', label = 'samples', zorder = 0)
            
            # text_box = ax.text(x.max(), y.max(), "Text in a Box", ha='left', va='top', bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})
            
            ax.text(x.max(), y.max(), str(round(num,2)), ha='left', va='bottom', 
                    bbox={'facecolor': cmap(corr), 'alpha': 1, 'pad': 5})
            
            # ax.set_xlabel(techs[i])
            # ax.set_ylabel(techs[j])
            
def solutions_heatmap2(techs, solutions, n_samples = 1000):
    import pandas
    import matplotlib
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    
    d = sample_in_hull(solutions, n_samples)
    
    d_df = pandas.DataFrame(d,
                            columns = techs)

    d_corr = d_df.corr()
    
    d_corr2 = d_corr + abs(d_corr.min().min())
    
    d_norm = d_corr2 / d_corr2.max().max()
    
    # Figure setup
    set_plot_options()
    plt.figure()
    n_vars = len(techs)
    fig, axs = plt.subplots(n_vars, n_vars, figsize = (20,10))
    fig.subplots_adjust(wspace = 0.4, hspace = 0.4)
    pad = 5
    
    cmap = matplotlib.cm.get_cmap('Spectral')
    
    for ax, col in zip(axs[0], techs):
        ax.set_title(col + '\n')
        
    for ax, row in zip(axs[:,0], techs):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
    
    
    # Fill figure
    for i in range(0,len(techs)):
        
        for j in range(0,len(techs)):
            
            ax = axs[j][i]
            
            ax.grid(False)
            
            corr = d_norm[techs[i]][techs[j]]
            num  = d_corr[techs[i]][techs[j]]
            
            # Turn off tick labels
            # ax.set_yticklabels([])
            # ax.set_xticklabels([])
            # ax.set_xticks([])
            # ax.set_yticks([])
            
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            ax.text(center_x, center_y, str(round(num,2)), 
                    fontsize = 20,
                    ha='center', va='center')
            
            color = cmap(corr)
            
            ax.set_facecolor(color)
            
            
def solutions_heatmap(techs, solutions, triangular = False, n_samples = 1000):
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
    
    
    
    
    
    
    