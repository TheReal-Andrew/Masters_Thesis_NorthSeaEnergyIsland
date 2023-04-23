import numpy as np
import matplotlib.pyplot as plt
import pypsa
import gorm as gm
import tim as tm
import pandas as pd

input_name = 'base0_opt.nc'
n = pypsa.Network(input_name)

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

n_opt = n.copy()
# Define custom constraints
def extra_functionalities(n, snapshots):
    gm.area_constraint(n, snapshots)
    gm.link_constraint(n, snapshots)

#%% Sensitivity analysis on marginal revenues
generators = ['P2X', 'Data']
sensitivity_cap = {key: None for key in generators}

for generator in generators:
    sensitivity_cap[generator] = {key: [] for key in n.generators.index}
    
marginal_cost_ranges = {
    'P2X': np.arange(1, 11, 0.05),
    'Data': np.arange(0.4, 1, 0.05)
}

for i, generator in enumerate(generators):
    
    for mc in marginal_cost_ranges[generator]:
        n.generators.loc[generator, 'marginal_cost'] = n_opt.generators.loc[generator, 'marginal_cost'] * mc

        # Solve
        n.lopf(pyomo=False,
               solver_name='gurobi',
               keep_shadowprices=True,
               keep_references=True,
               extra_functionality=extra_functionalities)
        
        # Store installed capacities
        for j in n.generators.index:
            sensitivity_cap[generator][j].append(n.generators.p_nom_opt[j])
            
        # Reset system to optimum system
        n = n_opt.copy()
            
#%% Plot
fig, axs = plt.subplots(nrows=1, ncols=len(generators), figsize=(6*len(generators),8))
for i,generator in enumerate(generators):
            
    for k in [s for s in n.generators.index if s in generators]:    
        axs[i].plot(marginal_cost_ranges[generator], sensitivity_cap[generator][k], label = k)
        axs[i].set_title(f'Optimal capacity for {generator}')
        axs[i].set_xlabel('Marginal cost')
        axs[i].set_ylabel('Capacity (p.u.)')
        axs[i].legend(loc='best')    

plt.tight_layout()
plt.show()

#%% Find intersection
from shapely.geometry import LineString

x = marginal_cost_ranges['P2X']
first_line = LineString(np.column_stack((x, sensitivity_cap['P2X']['Data'])))
second_line = LineString(np.column_stack((x, sensitivity_cap['P2X']['P2X'])))
intersection = first_line.intersection(second_line)
