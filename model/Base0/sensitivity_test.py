import numpy as np
import matplotlib.pyplot as plt
import modules.island_plt as ip
import pypsa
import gorm as gm
import tim as tm
import pandas as pd

ip.set_plot_options()

name = 'P2X'
step = 0.1
x = np.arange(0.5, 1.5 + step, step)
# x = np.linspace(0.2,1.8,17)

input_name = 'base0_opt.nc'
n = pypsa.Network(input_name)
n_opt = n.copy()

# Main parameter series
mp = tm.get_main_parameters()

# Main parameters
year        = 2030             # Choose year
r           = 0.07             # Discount rate
n_hrs       = 8760             # [hrs] Choose number of hours to simulate
DR          = 1.3              # Detour factor
wind_cap    = mp[year]['wind'] # [MW] Installed wind capacity
island_area = mp[year]['island_area']  # [m^2] total island area

link_efficiency = 3.5/(1000*100)   # Link efficency per km (per unit loss/km). 
link_sum_max    = wind_cap             # Total allowed link capacity
link_p_nom_min  = 0                    # Minimum allowed capacity for one link
link_limit      = float('inf')     # [MW] Limit links to countries. float('inf')

# Choose which countries to include of this list, comment unwanted out.
connected_countries =  [
                        "Denmark",         
                        "Norway",          
                        "Germany",         
                        "Netherlands",     
                        "Belgium",         
                        "United Kingdom"
                        ]

#%% ------- IMPORT DATA -----------------------------------

# ----- Wind capacity factor data ---------
wind_cf         = pd.read_csv(r'../../data/wind/wind_cf.csv',
                              index_col = [0], sep=",").iloc[:n_hrs,:]

# ----- Country demand and price ---------
# Import price and demand for each country for the year, and remove outliers
cprice, cload   = tm.get_load_and_price(year, connected_countries, n_std = 1)
cprice = cprice.iloc[:n_hrs,:] # Cut off to match number of snapshots
cload  = cload.iloc[:n_hrs,:]  # Cut off to match number of snapshots

# ----- Dataframe with bus data ---------
# Get dataframe with bus info, only for the connected countries.
bus_df          = tm.get_bus_df(connected_countries) # Import country data for linking
country_df      = bus_df[1:].copy() 

# ----- Dataframe with tech data ---------
tech_df         = tm.get_tech_data(year, r)

# ----- Area use data ---------
area_use        = tm.get_area_use()

# Add data to network for easier access when creating constraints
n.bus_df         = bus_df
n.area_use       = area_use
n.total_area     = island_area 
n.link_sum_max   = link_sum_max
n.link_p_nom_min = link_p_nom_min

# Add list of main links to network to differetniate
n.main_links = n.links[~n.links.index.str.contains("bus")].index

# Define custome constraints
def extra_functionalities(n, snapshots):
    gm.area_constraint(n, snapshots)
    gm.link_constraint(n, snapshots)

# Define generators and marginal cost ranges
generators = ['P2X', 'Data']
marginal_cost_ranges = {
    # 'P2X': np.arange(0.5, 1.6, 0.1),
    # 'Data': np.arange(0.5, 1.6, 0.1)
    'P2X': np.arange(1, 1.2, 0.1),
    'Data': np.arange(1, 1.2, 0.1)
}

sensitivity_cap = []
for generator in generators:
    for marginal_cost in marginal_cost_ranges[generator]:
        # Update generator marginal cost in network
        n.generators.loc[generator, 'marginal_cost'] = n_opt.generators.loc[generator, 'marginal_cost'] * marginal_cost
        
        # Solve network optimization problem
        n.lopf(pyomo=False, 
               solver_name='gurobi', 
               keep_shadowprices=True,
               keep_references=True, 
               extra_functionality=extra_functionalities)
        
        # Get optimal installed capacity and store it in list
        optimal_capacity = n.generators.loc[generator, 'p_nom_opt']
        sensitivity_cap.append(optimal_capacity)
        n.export_to_netcdf('sensitivity' + str(generator) + str(marginal_cost) + '.nc')

# Reshape sensitivity_cap list into a NumPy array
sensitivity_cap = np.array(sensitivity_cap).reshape((len(generators), len(marginal_cost_ranges[generators[0]])))

#%% Create a figure and plot the sensitivity analysis
fig, ax = plt.subplots()
for i, generator in enumerate(generators):
    ax[i].plot(marginal_cost_ranges[generator], sensitivity_cap[i, :], label=generator)
ax.set_xlabel('Marginal cost')
ax.set_ylabel('Installed capacity (MW)')
ax.legend()

plt.show()