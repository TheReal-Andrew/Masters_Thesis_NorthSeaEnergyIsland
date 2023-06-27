# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 09:29:02 2023

@author: lukas
"""

import os
import sys
# Add modules folder to path
os.chdir(os.path.join(os.path.dirname(__file__))) # Change working directory
sys.path.append(os.path.abspath('../../modules')) # Add modules to path

import pypsa
import pandas as pd
from ttictoc import tic, toc

import gorm as gm
import tim as tm
import pypsa_diagrams as pdiag
import matplotlib.dates as mdates
import os
import sys
os.chdir(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.abspath('../../modules'))
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pypsa
import gorm as gm
import tim as tm
import island_plt as ip
ip.set_plot_options()
# import pandas as pd
from matplotlib.ticker import (MultipleLocator)
from pytictoc import TicToc
t = TicToc() #create instance of time class


tic() # Start timer

gm.set_plot_options()

#%% ------- CONTROL -----------------------------------

# Main control
should_solve       = True
should_export      = True
should_plot        = False
should_bus_diagram = False
should_n_diagram   = False

# ---- User parameters - change this ------------------
project_name = 'base0_lonely_island'
year         = 2030             # Choose year

# Choose which countries to include of this list, comment unwanted out.
connected_countries =  [
                        # "Denmark",         
                        # "Norway",          
                        # "Germany",         
                        # "Netherlands",     
                        # "Belgium",         
                        # "United Kingdom"
                        ]

# Component control
add_storage  = True # Add storage on island
add_data     = True # Add datacenter on island
add_hydrogen = True # Add hydrogen production on island
add_c_gens   = True # Add country generators
add_c_loads  = True # Add country demand
add_moneybin = True

# ---- Main parameters loaded from tim module --------
mp, mp_gen = tm.get_main_parameters()

wind_cap     = mp[year]['wind']        # [MW] Installed wind capacity
island_area  = mp[year]['island_area'] # [m^2] total island area
r            = mp_gen['discount_rate'] # Discount rate
DR           = mp_gen['detour_factor'] # Detour factor

link_efficiency = 3.5/(1000*100)   # Link efficency per km (per unit loss/km). 
link_sum_max    = wind_cap         # Total allowed link capacity
link_p_nom_min  = 0                # Minimum allowed capacity for one link
link_limit      = float('inf')     # [MW] Limit links to countries. float('inf')

filename = f"/v_{year}_{project_name}_opt.nc" # Choose filename for export

#%% ------- IMPORT DATA -----------------------------------

# ----- Wind capacity factor data ---------
wind_cf         = pd.read_csv(r'../../data/wind/wind_cf.csv',
                       index_col = [0], sep=",")[-8760:]

# ----- Country demand and price ---------
# Import price and demand for each country for the year, and remove outliers
cprice, cload   = tm.get_load_and_price(year, connected_countries, n_std = 1)
cload = cload.applymap(lambda x: 0 if x < 0 else x) # Remove negative values

# ----- Dataframe with bus data ---------
# Get dataframe with bus info, only for the connected countries.
bus_df          = tm.get_bus_df(connected_countries) # Import country data for linking
country_df      = bus_df[1:].copy() 

# ----- Dataframe with tech data ---------
tech_df         = tm.get_tech_data(year, r)

# ----- Area use data ---------
area_use        = tm.get_area_use()

# ----- Leap year fix -----
# 2040 is a leap year, so has 8784 hours. 2041 is used for the snapshots if the 
# year is 2040, so there are still 8760 hours.
if year == 2040:
    add = 1
else:
    add = 0

#%% ------- NETWORK -----------------------------------------------------------

# ----- initialize network ---------
n = pypsa.Network()
t = pd.date_range(f'{year+add}-01-01 00:00', f'{year+add}-12-31 23:00', freq = 'H')
n.set_snapshots(t)

# Add data to network for easier access when creating constraints
n.bus_df         = bus_df
n.area_use       = area_use
n.total_area     = island_area 
# n.link_sum_max   = link_sum_max
# n.link_p_nom_min = link_p_nom_min
n.filename       = filename
n.connected_countries = connected_countries

# ----- Add buses-------------------
# Add an island bus and a bus for each country.
# Add multiple buses by passing arrays from bus_df to parameters and using madd
n.madd('Bus',
       names = bus_df['Bus name'], 
       x     = bus_df['X'].values,
       y     = bus_df['Y'].values,
        )


# ----- Add links -------------------
# For each country, add a link from the island to the country and from the
# country to the island. These will act as bidirectional links when the 
# capacities are tied together with the extra_functionality function 
# "marry_link".

for country in country_df['Bus name']:
    
    # Get link distance in [km]
    distance = gm.get_earth_distance(bus_df.loc['Energy Island']['X'],
                                          country_df.loc[country]['X'],
                                          bus_df.loc['Energy Island']['Y'],
                                          country_df.loc[country]['Y'])
    
    # Define capital cost (cc) and efficiency (eff) depending on link distance
    # cc  = tech_df['capital cost']['link'] * distance * DR
    # eff = 1 - (link_efficiency * distance * DR)
    
    # Add two links using madd. Efficiency is applied to both links, while
    # capital cost and carrier for MAA is applied to only one.
    # n.madd('Link',
    #         names            = ["Island_to_" + country,    country + "_to_Island"],
    #         bus0             = ['Energy Island',           country],
    #         bus1             = [country,                   'Energy Island'],
    #         carrier          = ['link_' + country,         ''],
    #         p_nom_extendable = [True,                      True],
    #         capital_cost     = [cc,                        0],
    #         efficiency       = [eff,                       eff],
    #         )

# Add the main links as network variable, to access in extra_functionality
# constraints.
# n.main_links = n.links.loc[n.links.bus0 == "Energy Island"].index

# ______________________________ ### COUNTRIES ### ____________________________

# ----- Add generators for countries--------------------
# Add generators to each country bus with marginal costs equal to electricity
# price in each timestep. 
if add_c_gens:
    for country in country_df['Bus name']:
        n.add('Generator',
              name             = "Gen " + country,
              bus              = country,
              capital_cost     = 0,
              marginal_cost    = cprice[country_df.loc[country]['Abbreviation']].values,
              p_nom_extendable = True
              )
        
# ----- Add loads for countries--------------------
# Add loads to each country bus, equivalent to the predicted demands for the
# year according to PES30P
if add_c_loads:
    for country in country_df['Bus name']:
        n.add('Load',
              name    = "Load " + country,
              bus     = country,
              p_set   = cload[country_df.loc[country]['Abbreviation']].values,
              ) 

# ______________________________ ### ISLAND ### _______________________________

# ----- Add wind generator --------------------
# Add a generator to represent offshore wind farms, with capacity according
# to the year. P_nom_extendable = True with p_nom_min and max set to the same
# value forces the model to built the capacity, but there is not capital cost.
n.add("Generator",
      "Wind",
      bus               = bus_df.loc['Energy Island']['Bus name'], # Add to island bus
      carrier           = "wind",
      p_nom_extendable  = True,
      p_nom_max         = wind_cap, # Ensure that capacity is pre-built
      p_nom_min         = wind_cap, # Ensure that capacity is pre-built
      p_max_pu          = wind_cf['electricity'].values,
      marginal_cost     = tech_df['marginal cost']['wind turbine'],
       )

# ----- Add battery storage --------------------
# Add storage to the island, represented by lithium-ion battery. Cyclic.
if add_storage:
    n.add("Store",
          "Storage",
          bus               = bus_df.loc['Energy Island']['Bus name'], # Add to island bus
          carrier           = "Storage",
          e_nom_extendable  = True,
          e_cyclic          = True,
          capital_cost      = tech_df['capital cost']['storage'],
          marginal_cost     = tech_df['marginal cost']['storage']
          )

# ----- Add hydrogen production --------------------
# Add hydrogen production. The generator is made negative by setting p_max_pu
# to 0, and p_min_pu to -1. This generator will produce "negative" energy, 
# which is effectively a load, but can be optimized. Since the energy is 
# negative, the marginal cost becomes a gain for the system (marginal revenue).
# This doesn't affect the capital cost.
if add_hydrogen:
    n.add("Generator",
          "P2X",
          bus               = bus_df.loc['Energy Island']['Bus name'], # Add to island bus
          carrier           = "P2X",
          p_nom_extendable  = True,
          p_max_pu          = 0,
          p_min_pu          = -1,
          capital_cost      = tech_df['capital cost']['hydrogen'],
          marginal_cost     = tech_df['marginal cost']['hydrogen'],
          )

# ----- Add datacenter --------------------
# Add datacenter, which consumes power to produce value. Made as negative 
# generator in the same way as hydrogen, except p_max_pu and p_min_pu are both
# effectively -1. This ensures a constant "load" that can be determined by the
# optimization, since a datacenter requires constant power supply.
if add_data:
    n.add("Generator",
            "Data",
            bus               = bus_df.loc['Energy Island']['Bus name'], # Add to island bus
            carrier           = "Data",
            p_nom_extendable  = True,
            p_max_pu          = -0.99,
            p_min_pu          = -1,
            capital_cost      = tech_df['capital cost']['datacenter'],
            marginal_cost     = tech_df['marginal cost']['datacenter'],
            )

# ----- Add moneybin --------------------
# The moneybin is a virtual component. Since the negative generators use power
# to reduce the objective function value, this could result in a negative 
# objective function value, which is not compatible with MAA. This moneybin
# ensures that the objective function is always positive. Must be removed
# from MAA slack calculation in MAA script. 
# The moneybin is set up as a generator that is forced to build 1 [MW] capacity
# but with so high marginal cost it will never be utilized in the model.
if add_moneybin:
    n.add("Generator",
          "MoneyBin",
          bus               = bus_df.loc['Energy Island']['Bus name'], # Add to island bus
          carrier           = "MoneyBin",
          p_nom_extendable  = True,
          p_nom_min         = 1, # Always build 1 [MW]
          p_nom_max         = 1, # Always build 1 [MW]
          capital_cost      = island_area/n.area_use['data']*tech_df['marginal cost']['datacenter']*8760,
          marginal_cost     = island_area/n.area_use['data']*tech_df['marginal cost']['datacenter']*8760,
          )

# %% Extra functionality
def extra_functionalities(n, snapshots):
    
    gm.area_constraint(n, snapshots) # Add an area constraint
    
    # gm.link_constraint(n, snapshots) # Add constraint on the link capacity sum
    
    # gm.marry_links(n, snapshots) # Add constraint to link capacities of links

#%% Solve
if should_solve:
    n.lopf(
           pyomo = False,
           solver_name = 'gurobi', 
           keep_shadowprices = True, # Keep dual-values
           keep_references = True,   
           extra_functionality = extra_functionalities,
           )
    #%%
    if should_export:
        export_path = os.getcwd() + filename
        n.export_to_netcdf(export_path)
    
#%% Plot

if should_plot:
    gm.plot_geomap(n)

if should_n_diagram:
    
    It = 'Island to '
    
    index2 = [
              It+'Germany',
              It+'Norway',
              It+'Denmark',
              It+'Netherlands',
              It+'Belgium',
              It+'United Kingdom',
              ]
    
    pdiag.draw_network(n, spacing = 1, handle_bi = True, pos = None,
                        index1 = index2,
                       line_length = 2,
                       filename = 'graphics/pypsa_diagram_3_2.pdf')
    
if should_bus_diagram:
    pdiag.draw_bus(n, 'Energy Island', 
                   # bus_color = 'azure',
                   handle_bi = True, 
                   link_line_length = 1.1,
                   filename = 'graphics/bus_diagram1.pdf')

#%% Finish message

print(f'\n ### Runtime: {round(toc(), 2)} seconds \n')

#%% Finish message
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), dpi=300, constrained_layout=True)

# Set limits and formatter for the x-axis (time)
axs.set_xlim([n.stores_t.p['Storage'].index[0], n.stores_t.p['Storage'].index[-1]])
axs.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

# Set labels and title for the plot
axs.set_xlabel('Time [day]')
axs.set_ylabel('Energy level [GWh]')
axs.set_title('Lonely island storage power flow in ' + str(year))

# Set limits, ticks, and a horizontal line at zero for the original y-axis (dispatch)
axs.set_ylim([-1_200, 1_500])
axs.set_yticks(np.arange(-1_200, 1_800, 300), [-1.2, -0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9, 1.2, 1.5])
axs.axhline(y=0, linestyle="--", color="k")
axs.set_ylabel('Dispatch [GW]')

# Plot dispatch on the original y-axis
axs.plot(n.stores_t.p['Storage'].resample('D').mean(), color='#8da0cb', label="Dispatch")

# Adjust the layout for better appearance
plt.tight_layout()

# Save the figure to a PNG file
plt.savefig('lonely_island_storage_powerflow_' + str(year) + '.png', format='png', bbox_inches='tight')


#%%
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), dpi=300, constrained_layout=True)

# Set labels and title for the plot
axs.set_xlabel('Dispatch [GW]')
axs.set_ylabel('Frequency')
axs.set_title('Dispatch histogram for the lonely island in ' + str(year))

# Create histogram bins based on the dispatch data
dispatch_data = n.stores_t.p['Storage'].resample('D').mean() / 1000  # Divide by 1000 for GW
bins = 50

# Plot the histogram of dispatch with added spacing between bins
axs.hist(dispatch_data, bins=bins, color='#8da0cb', rwidth=0.8)

# Add a vertical line at 0
axs.axvline(x=0, linestyle="--", color="k")

# Add text annotations for charging and discharging
axs.text(0.05, 0.9, 'Charging', transform=axs.transAxes, ha='left')
axs.text(0.95, 0.9, 'Discharging', transform=axs.transAxes, ha='right')

# Set the x-axis major ticks as even and round numbers
tick_spacing = 0.5  # Set the desired tick spacing
x_ticks_major = np.arange(np.floor(np.min(dispatch_data) / tick_spacing) * tick_spacing,
                          np.ceil(np.max(dispatch_data) / tick_spacing) * tick_spacing + tick_spacing,
                          tick_spacing)
axs.set_xticks(x_ticks_major)

# Set the x-axis minor ticks for every bin
minor_tick_locations = np.arange(np.floor(np.min(dispatch_data)), np.ceil(np.max(dispatch_data)) + 1, 0.1)
axs.set_xticks(minor_tick_locations, minor=True)

# Set the x-axis limit
axs.set_xlim([-1, 1.5])

# Adjust the layout for better appearance
plt.tight_layout()

# Save the figure to a PNG file
plt.savefig('dispatch_histogram_' + str(year) + '.png', format='png', bbox_inches='tight')

#%%
colors = gm.get_color_codes()

# Performing the calculation
wind = n.generators_t.p['Wind'].resample('D').mean()

# Generate data
storage_data = n.stores_t.p['Storage'].resample('D').mean()
p2x_data = n.generators_t.p['P2X'].resample('D').mean()
data_data = n.generators_t.p['Data'].resample('D').mean()

# Create a new figure
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5), dpi=300, constrained_layout=True)

# Create a stacked plot
ax.stackplot(storage_data.index, -data_data, -p2x_data, -storage_data,colors=[colors['Data'],colors['P2X'],colors['Storage']] ,labels=['IT', 'P2X', 'Storage'])
ax.set_xlim([n.stores_t.p['Storage'].index[0], n.stores_t.p['Storage'].index[-1]])
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.plot(wind, label="Wind power", color = "k")

# Add title and labels
ax.set_title('Stacked power plot for lonely island in ' + str(year))
ax.set_xlabel('Time [Day]')
ax.set_ylabel('Power [MW]')

# Add a legend below the plot
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=5)

# Adjust the layout to make room for the legend
plt.subplots_adjust(bottom=0.2)

# Adjust the layout for better appearance
plt.tight_layout()

# Save the figure to a PNG file
plt.savefig('stacked_lonely_' + str(year) + '.pdf', format='pdf', bbox_inches='tight')


#%%
print('Length of data_data:', len(data_data))
print('Length of p2x_data:', len(p2x_data))
print('Length of storage_data:', len(storage_data))
print('Length of wind:', len(wind))


















