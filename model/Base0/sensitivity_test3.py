import os
import sys
# Add modules folder to path
sys.path.append(os.path.abspath('../../modules'))
sys.path.append(os.path.abspath('../../modules'))
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pypsa
import gorm as gm
import tim as tm
import pandas as pd
from matplotlib.ticker import (MultipleLocator)
from shapely.geometry import LineString

input_name = 'base0_opt.nc'
n = pypsa.Network(input_name)
n_opt = n.copy()

# Define custom constraints
def extra_functionalities(n, snapshots):
    gm.area_constraint(n, snapshots)
    gm.link_constraint(n, snapshots)

#%% Sensitivity analysis on marginal revenues
components = ['P2X', 'Data', 'Island_store']

cc_ranges = {
    'P2X': np.arange(0,1.1,0.1),
    'Data': np.arange(1,2.1,0.1),
    'Island_store': np.arange(0, 1.1, 0.1),
}

mc_ranges = {
    'P2X': np.append([1,2.4],np.arange(2.5,10.1,0.1)),
    'Data': np.append(np.append([0,0.5],np.arange(0.501,0.602,0.001)),[0.7,1]),
    'Island_store': np.arange(0, 2.1, 0.1),
}

#%% Initialize/reset dictionaries
cc_sensitivity_cap = {key: None for key in components}
mc_sensitivity_cap = {key: None for key in components}

for component in components:
    cc_sensitivity_cap[component] = {key: [] for key in (list(n.generators.index) + list(n.stores.index))}
    mc_sensitivity_cap[component] = {key: [] for key in (list(n.generators.index) + list(n.stores.index))}

#%% Sensitivity sweep: capital cost

for i, component in enumerate(components):
    
    for cc in cc_ranges[component]:
        
        # Basic PyPSA setup
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
        
        
        # Run & store
        if component == 'Island_store':
            n.stores.loc[component, 'capital_cost'] = n.stores.loc[component, 'capital_cost'] * cc
        else:
            n.generators.loc[component, 'capital_cost'] = n_opt.generators.loc[component, 'capital_cost'] * cc

        # Solve
        n.lopf(pyomo=False,
               solver_name='gurobi',
               keep_shadowprices=True,
               keep_references=True,
               extra_functionality=extra_functionalities)
        
        # Store installed capacities
        for j in n.generators.index:
            cc_sensitivity_cap[component][j].append(n.generators.p_nom_opt[j])
        cc_sensitivity_cap[component]['Island_store'].append(n.stores.loc['Island_store'].e_nom_opt)
        
#%% Sensitivity sweep: marginal cost
for i, component in enumerate(components):
    
    for mc in mc_ranges[component]:
        
        # Basic PyPSA setup
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
        
        
        # Run & store
        if component == 'Island_store':
            n.stores.loc[component, 'capital_cost'] = n.stores.loc[component, 'capital_cost'] * mc
        else:
            n.generators.loc[component, 'capital_cost'] = n_opt.generators.loc[component, 'capital_cost'] * mc

        # Solve
        n.lopf(pyomo=False,
               solver_name='gurobi',
               keep_shadowprices=True,
               keep_references=True,
               extra_functionality=extra_functionalities)
        
        # Store installed capacities
        for j in n.generators.index:
            mc_sensitivity_cap[component][j].append(n.generators.p_nom_opt[j])
        mc_sensitivity_cap[component]['Island_store'].append(n.stores.loc['Island_store'].e_nom_opt)
        
#%% Save data in file
# save dictionary to person_data.pkl file

with open('cc_sensitivity_cap.pkl', 'wb') as fp:
    pickle.dump(cc_sensitivity_cap, fp)
    # print('ensitivity_cap saved successfully to file')
    
with open('mc_sensitivity_cap.pkl', 'wb') as fp:
    pickle.dump(mc_sensitivity_cap, fp)
    # print('store_cap saved successfully to file')

#%% Read data from file
# Read dictionary pkl file
with open('cc_sensitivity_cap.pkl', 'rb') as fp:
    cc_sensitivity_cap = pickle.load(fp)
    
with open('mc_sensitivity_cap.pkl', 'rb') as fp:
    mc_sensitivity_cap = pickle.load(fp)
    
#%% Plot capital cost sweep
fig, axs = plt.subplots(nrows=1, ncols=len(components), figsize=(10*len(components),7), dpi = 300)

for i, component in enumerate(components):
    x = cc_ranges[component]
        
    for k in [s for s in (list(n.generators.index) + list(n.stores.index)) if s in components]:
        y = cc_sensitivity_cap[component][k]
        axs[i].plot(x, y, label = k)
        axs[i].set_title(f'Sensitivity of {component} capital cost')
        axs[i].set_xlabel('Capital cost coefficient [-]')
        axs[i].set_ylabel('Installed capacity [MW]')
        axs[i].legend(loc='best')
        
        # axs[i].set_yticks(np.arange(0,4000*(1+0.1),500))
        
        # if component == "P2X":
        #     axs[i].set_xlim([min(x),max(x)])
        #     axs[i].set_ylim([-50,4000])

        #     axs[i].set_xticks(np.arange(min(x),max(x)*(1),max(x)/10))
        
        #     axs[i].xaxis.set_minor_locator(MultipleLocator(max(x)/100))
        #     axs[i].yaxis.set_minor_locator(MultipleLocator(100))
        # else:
            
        #     axs[i].set_xlim([0.5,0.6])
        #     axs[i].set_ylim([-50,4000])
            
        #     axs[i].set_xticks(np.arange(0.5,0.61,0.1/10))
            
        #     axs[i].xaxis.set_minor_locator(MultipleLocator(max(x)/1000))
        #     axs[i].yaxis.set_minor_locator(MultipleLocator(100))
            
        # axs[i].tick_params(which='minor')

        # first_line = LineString(np.column_stack((mc_ranges[component], cc_sensitivity_cap[component]['Data'])))
        # second_line = LineString(np.column_stack((mc_ranges[component], cc_sensitivity_cap[component]['P2X'])))
        # idx = first_line.intersection(second_line)
        # axs[i].plot(idx.x,idx.y,'ro')
        
        # if component == "P2X":
        #     axs[i].text(idx.x*(1+0.05), idx.y*(1+0.05), 
        #                 "Coefficient: " + str(round(idx.x,2)) + "\n" + str(component) + " Capital cost: " 
        #                 + str(round(n.generators.loc[component, 'capital_cost']*idx.x)) 
        #                 + " €"
        #                 + "\nCapacity: " + str(round(idx.y,1)) + " MW")
        # else: 
        #     axs[i].text(idx.x*(1+0.005), idx.y*(1+0.005), 
        #                 "Coefficient: " + str(round(idx.x,2)) + "\n" + str(component) + " revenue: " 
        #                 + str(round(n.generators.loc[component, 'capital_cost']*idx.x)) 
        #                 + " €"
        #                 + "\nCapacity: " + str(round(idx.y,1)) + " MW")

plt.tight_layout()
plt.show()

#%% Plot marginal cost sweep
fig, axs = plt.subplots(nrows=1, ncols=len(components), figsize=(10*len(components),7), dpi = 300)

for i, component in enumerate(components):

    x = mc_ranges[component]
    print('Vi er i gang med: ' + component)    
    for k in [s for s in (list(n.generators.index) + list(n.stores.index)) if s in components]:
        print('Så kommer: ' + k)  
        y = mc_sensitivity_cap[component][k]
        axs[i].plot(x, y, label = k)
        axs[i].set_title(f'Sensitivity sweep of {component} marginal cost/revenue')
        axs[i].set_xlabel('Marginal coefficient [-]')
        axs[i].set_ylabel('Installed capacity [MW]')
        axs[i].legend(loc='best')
        
        axs[i].set_yticks(np.arange(0,4000*(1+0.1),500))
        
        if component == "P2X":
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

        # first_line = LineString(np.column_stack((mc_ranges[component], cc_sensitivity_cap[component]['Data'])))
        # second_line = LineString(np.column_stack((mc_ranges[component], cc_sensitivity_cap[component]['P2X'])))
        # idx = first_line.intersection(second_line)
        # axs[i].plot(idx.x,idx.y,'ro')
        
        # if component == "P2X":
        #     axs[i].text(idx.x*(1+0.05), idx.y*(1+0.05), 
        #                 "Coefficient: " + str(round(idx.x,2)) + "\n" + str(component) + " Capital cost: " 
        #                 + str(round(n.generators.loc[component, 'capital_cost']*idx.x)) 
        #                 + " €"
        #                 + "\nCapacity: " + str(round(idx.y,1)) + " MW")
        # else: 
        #     axs[i].text(idx.x*(1+0.005), idx.y*(1+0.005), 
        #                 "Coefficient: " + str(round(idx.x,2)) + "\n" + str(component) + " revenue: " 
        #                 + str(round(n.generators.loc[component, 'capital_cost']*idx.x)) 
        #                 + " €"
        #                 + "\nCapacity: " + str(round(idx.y,1)) + " MW")

plt.tight_layout()
plt.show()