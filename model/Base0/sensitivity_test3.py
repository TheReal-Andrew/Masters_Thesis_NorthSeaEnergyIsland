import os
import sys
# Add modules folder to path
sys.path.append(os.path.abspath('../../modules')) 
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pypsa
import gorm as gm
import tim as tm
import pandas as pd

input_name = 'base0_opt.nc'
n = pypsa.Network(input_name)
n_opt = n.copy()

# Define custom constraints
def extra_functionalities(n, snapshots):
    gm.area_constraint(n, snapshots)
    gm.link_constraint(n, snapshots)

#%% Sensitivity analysis on marginal revenues
generators = ['P2X', 'Data']

marginal_cost_ranges = {
    # 'P2X': np.arange(1, 3, 1),
    'P2X': np.append([1,2.4],np.arange(2.5,10.1,0.1)),
    'Data': np.append(np.append([0,0.5],np.arange(0.501,0.602,0.001)),[0.7,1])
}

#%% Initialize/reset dictionaries
sensitivity_cap = {key: None for key in generators}
store_cap = {key: None for key in generators}

for generator in generators:
    sensitivity_cap[generator] = {key: [] for key in n.generators.index}

for generator in generators:
    store_cap[generator] = []

#%% Sensitivity loop

for i, generator in enumerate(generators):
    
    for mc in marginal_cost_ranges[generator]:
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

        # ------- IMPORT DATA -----------------------------------

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
        store_cap[generator].append(n.stores.loc["Island_store"].e_nom_opt)
        
#%% Save data in file
# save dictionary to person_data.pkl file

with open('sensitivity_rev_gen.pkl', 'wb') as fp:
    pickle.dump(sensitivity_cap, fp)
    # print('ensitivity_cap saved successfully to file')
    
with open('store_rev_cap.pkl', 'wb') as fp:
    pickle.dump(store_cap, fp)
    # print('store_cap saved successfully to file')

#%% Read data from file
# Read dictionary pkl file
with open('sensitivity_rev_gen.pkl', 'rb') as fp:
    sensitivity_cap = pickle.load(fp)
    
with open('store_rev_cap.pkl', 'rb') as fp:
    store_cap = pickle.load(fp)
    
#%% Plot
fig, axs = plt.subplots(nrows=1, ncols=len(generators), figsize=(10*len(generators),7), dpi = 300)
from matplotlib.ticker import (MultipleLocator)
# import modules.island_plt as ip
# ip.set_plot_options()
from shapely.geometry import LineString

for i,generator in enumerate(generators):
    x = marginal_cost_ranges[generator]
    axs[i].plot(x, store_cap[generator], label = "Store")
        
    for k in [s for s in n.generators.index if s in generators]:
        y = sensitivity_cap[generator][k]
        axs[i].plot(x, y, label = k)
        axs[i].set_title(f'Sensitivity of {generator} marginal revenue')
        axs[i].set_xlabel('Marginal revenue coefficient [-]')
        axs[i].set_ylabel('Installed capacity [MW]')
        axs[i].legend(loc='best')
        
        axs[i].set_yticks(np.arange(0,4000*(1+0.1),500))
        
        if generator == "P2X":
            axs[i].set_xlim([min(x),max(x)])
            axs[i].set_ylim([-50,4000])

            axs[i].set_xticks(np.arange(min(x),max(x)*(1),max(x)/10))
        
            axs[i].xaxis.set_minor_locator(MultipleLocator(max(x)/100))
            axs[i].yaxis.set_minor_locator(MultipleLocator(100))
        else:
            
            axs[i].set_xlim([0.5,0.6])
            axs[i].set_ylim([-50,4000])
            
            axs[i].set_xticks(np.arange(0.5,0.61,0.1/10))
            
            axs[i].xaxis.set_minor_locator(MultipleLocator(max(x)/1000))
            axs[i].yaxis.set_minor_locator(MultipleLocator(100))
            
        axs[i].tick_params(which='minor')

        first_line = LineString(np.column_stack((marginal_cost_ranges[generator], sensitivity_cap[generator]['Data'])))
        second_line = LineString(np.column_stack((marginal_cost_ranges[generator], sensitivity_cap[generator]['P2X'])))
        idx = first_line.intersection(second_line)
        axs[i].plot(idx.x,idx.y,'ro')
        
        if generator == "P2X":
            axs[i].text(idx.x*(1+0.05), idx.y*(1+0.05), 
                        "Coefficient: " + str(round(idx.x,2)) + "\n" + str(generator) + " revenue: " 
                        + str(round(n.generators.loc[generator, 'marginal_cost']*idx.x)) 
                        + " €"
                        + "\nCapacity: " + str(round(idx.y,1)) + " MW")
        else: 
            axs[i].text(idx.x*(1+0.005), idx.y*(1+0.005), 
                        "Coefficient: " + str(round(idx.x,2)) + "\n" + str(generator) + " revenue: " 
                        + str(round(n.generators.loc[generator, 'marginal_cost']*idx.x)) 
                        + " €"
                        + "\nCapacity: " + str(round(idx.y,1)) + " MW")

plt.tight_layout()
plt.show()

