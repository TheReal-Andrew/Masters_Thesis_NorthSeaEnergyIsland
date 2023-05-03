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
    lhs = linexpr(
                   (n.area_use['hydrogen'], vars_gen["P2X"]), 
                   (n.area_use['data'],     vars_gen["Data"]), 
                   (n.area_use['storage'],  vars_store['Island_store'])
                  )
    
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
    
def solutions_2D(techs, solutions,
                 optimal_solutions = None,
                 tech_titles = None,
                 n_samples = 1000,
                 title = 'MAA_plot',
                 plot_samples = False,
                 plot_heatmap = True,
                 cmap = 'Blues',
                 bins = 25,
                 filename = None,
                 alpha = 1):
    # Take a multi-dimensional MAA polyhedron, and plot each "side" in 2D.
    # Plot the polyhedron shape, samples within and correlations.
    import pandas
    import matplotlib
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    import numpy as np
    from scipy.stats import gaussian_kde
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    
    pad = 5
    n_cols = len(techs)
    
    if tech_titles == None: 
        tech_titles = techs
    
    # Sample polyhedron
    d = sample_in_hull(solutions, n_samples)
    
    # -------- create correlation matrix --------------------------
    # Create dataframe from samples
    d_df = pandas.DataFrame(d, columns = techs)
    
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
    
    # Set titles
    for ax, col in zip(axs[0], tech_titles):
        ax.set_title(col + '\n')
    
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
        
        d_df[techs[j]].hist(bins = 50, ax = ax,
                            color = 'tab:purple', rwidth = 0.9,
                            label = 'histogram')
        
        ax.text(0.5, text_lift, 'Histogram', ha='center', va='top', 
                transform=ax.transAxes, fontsize = 16, color = 'gray')
    
    
    # lower traingle of subplots
    for j in range(0, len(techs)):
        for i in range(0, j):
            
            ax = axs[j][i]
            
            ax.text(0.5, text_lift, 'Scatter plot with vertices', ha='center', va='top',
                    transform=ax.transAxes, fontsize=16, color = 'gray')
            
            # MAA solutions
            x, y = solutions[:,i],   solutions[:,j]
            
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
                l0, = ax.plot(solutions[simplex, i], solutions[simplex, j], 'k-', 
                        label = 'faces', zorder = 0)
                
            # Plot vertices from solutions
            l1, = ax.plot(x, y,
                      'o', label = "Near-optimal",
                      color = 'lightcoral', zorder = 2)
            
            # list of legend handles and labels
            l_list, l_labels   = [l0, l1, hb], ['Polyhedron face', 'Near-optimal MAA points', 'Sample density']
            
            # optimal solutions
            if not optimal_solutions == None:
                x_opt, y_opt = optimal_solutions[i],   optimal_solutions[j]
                
                # Plot optimal solutions
                l2, = ax.plot(x_opt, y_opt,
                          'o', label = "Optimal", 
                          ms = 20, color = 'red',
                          zorder = 3)
                
                l_list.append(l2)
                l_labels.append('Optimal solution')
    
    # Place legend below subplots
    ax = axs[len(techs)-1, int(np.median([1,2,3]))-1] # Get center axis
    ax.legend(l_list,
              l_labels, 
              loc = 'center', ncol = 3,
              bbox_to_anchor=(0.5, -0.25),fancybox=False, shadow=False,)
    
    fig.suptitle(title, fontsize = 24)
    
    if not filename == None:
        fig.savefig(filename, format = 'pdf', bbox_inches='tight')
        
def solutions_3D(techs, solutions,
                 figsize = (10,10),
                 markersize = 7, linewidth = 3,
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
    
def bake_local_area_pie(n, plot_title, exportname = None, ax = None):
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
    Store_a = n.stores.e_nom_opt["Island_store"] * n.area_use['storage']
    
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
    
    ax.set_title(plot_title,
              fontsize = 16,
              pad = 20)
    
    if not labels == None:
       ax.legend(labels = labels, 
                 loc = 'center', ncol = 3,
                 bbox_to_anchor=(0.5, -0.1), fancybox=False, shadow=False,)
    
    if not exportname == None and ax == None:
        fig.savefig(exportname, format = 'pdf', bbox_inches='tight')
        
def bake_capacity_pie(n, plot_title, exportname = None, ax = None):
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
    store_capacity = n.stores.e_nom_opt["Island_store"] # [MWh]
    
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
    
    ax.set_title(plot_title,
              fontsize = 16,
              pad = 20)
    
    if not labels == None:
       ax.legend(labels = labels, 
                 loc = 'center', ncol = 4,
                 bbox_to_anchor=(0.5, -0.1), fancybox=False, shadow=False,)
    
    if not exportname == None:
        fig.savefig(exportname, format = 'pdf', bbox_inches='tight')
    
    
    
    
    
    
    