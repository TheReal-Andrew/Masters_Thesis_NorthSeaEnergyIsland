# -*- coding: utf-8 -*-
"""
Created on Sat May  6 15:56:27 2023

@author: lukas
"""

import os
import sys
# Add modules folder to path
os.chdir(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.abspath('../../modules')) 

import pypsa
import pandas as pd

import gorm as gm
import tim as tm
import pypsa_diagrams as pdiag

#%% 

# ---- User parameters - change this ------------------
project_name = 'links_test'
year         = 2030             # Choose year

connected_countries =  [
                        "Denmark",         
                        "Norway",          
                        "Germany",         
                        "Netherlands",     
                        "Belgium",         
                        "United Kingdom"
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
link_sum_max    = 3000         # Total allowed link capacity

filename = f"/v_{year}_{project_name}_opt.nc" # Choose filename for export

#%% ------- IMPORT DATA -----------------------------------

# ----- Wind capacity factor data ---------
wind_cf         = pd.read_csv(r'../../data/wind/wind_cf.csv',
                       index_col = [0], sep=",")[:8760]

# ----- Country demand and price ---------
# Import price and demand for each country for the year, and remove outliers
cprice, cload   = tm.get_load_and_price(year, connected_countries, n_std = 1)
cload = cload.applymap(lambda x: 0 if x < 0 else x)

#Create dataframe with info on buses. Names, x (Longitude) and y (Latitude) 
bus_df = tm.get_bus_df(connected_countries)
country_df      = bus_df[1:].copy() 

# ----- Dataframe with tech data ---------
tech_df         = tm.get_tech_data(year, r)

# ----- Area use data ---------
area_use        = tm.get_area_use()
 
#%% ------- NETWORK -----------------------------------------------------------

# ----- initialize network ---------
n = pypsa.Network()
t = pd.date_range('2030-01-01 00:00', '2030-12-31 23:00', freq = 'H')
n.set_snapshots(t)

#%%

# Add data to network for easier access when creating constraints
n.bus_df         = bus_df
n.area_use       = area_use
n.total_area     = island_area 
n.link_sum_max   = link_sum_max
n.filename       = filename


# ----- Add buses-------------------
# Add multiple buses by passing arrays from bus_df to parameters and using madd
n.madd('Bus',
       names = bus_df['Bus name'], 
       x     = bus_df['X'].values,
       y     = bus_df['Y'].values,
        )

#%% Add links -----------------------------------------------------

# # ----- Add links--------------------
# for country in country_df['Bus name']:
    
#     # Get link distance in [km]
#     distance = gm.get_earth_distance(bus_df.loc['Energy Island']['X'],
#                                      country_df.loc[country]['X'],
#                                      bus_df.loc['Energy Island']['Y'],
#                                      country_df.loc[country]['Y'])
    
#     # Add bidirectional link with loss and marginal cost
#     gm.add_bi_link(n,
#                     bus0          = bus_df.loc['Energy Island']['Bus name'],   # From Energy island
#                     bus1          = country,                        # To country bus
#                     link_name     = "Island to " + country,         # Link name
#                     efficiency    = 1, #- (link_efficiency * distance * DR),
#                     capital_cost  = tech_df['capital cost']['link'] * distance * DR,
#                     marginal_cost = tech_df['marginal cost']['link'],
#                     carrier       = 'link_' + country,
#                     p_nom_extendable = True,
#                     )
    
# # Add list of main links to network to differetniate
# n.main_links = n.links[~n.links.index.str.contains("bus")].index

# for country in country_df['Bus name']:
    
#     # Get link distance in [km]
#     distance = gm.get_earth_distance(bus_df.loc['Energy Island']['X'],
#                                           country_df.loc[country]['X'],
#                                           bus_df.loc['Energy Island']['Y'],
#                                           country_df.loc[country]['Y'])
        
#     n.madd('Link',
#            names = ["Island_to_" + country,    country + "_to_Island"],
#            bus0  = ['Energy Island',           country],
#            bus1  = [country,                   'Energy Island']
#            )

# #List of link destionations from buses
link_destinations = n.buses.index.values

DR = 1.2
j = 0
for i in link_destinations[1:]:         #i becomes each string in the array
    j = j + 1
    
    distance = gm.get_earth_distance(bus_df.loc['Energy Island']['X'],
                                      country_df.loc[i]['X'],
                                      bus_df.loc['Energy Island']['Y'],
                                      country_df.loc[i]['Y'])
    
    n.add(                        #Add component
        "Link",                         #Component type
        "Island_to_" + i,               #Component name
        bus0             = "Energy Island",    #Start Bus
        bus1             = i,           #End Bus
        carrier          = "link_" + i,        #Define carrier type
        p_min_pu         = -1,          #Make links bi-directional
        p_nom            = 0,           #Power capacity of link
        p_nom_extendable = True,        #Extendable links
        capital_cost     = tech_df['capital cost']['link'] * distance * DR,
        marginal_cost    = tech_df['marginal cost']['link'],
        )
    
n.main_links = n.links.index

# ______________________________ ### COUNTRIES ### ____________________________
# ----- Add generators for countries--------------------
#Add generators to each country bus with varying marginal costs
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
# Add loads to each country bus
if add_c_loads:
    for country in country_df['Bus name']:
        n.add('Load',
              name    = "Load " + country,
              bus     = country,
              p_set   = cload[country_df.loc[country]['Abbreviation']].values,
              ) 

# ______________________________ ### ISLAND ### _______________________________
# ----- Add wind generator --------------------
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
if add_storage:
    n.add("Store",
          "Island_store",
          bus               = bus_df.loc['Energy Island']['Bus name'], # Add to island bus
          carrier           = "Store1",
          e_nom_extendable  = True,
          e_cyclic          = True,
          capital_cost      = tech_df['capital cost']['storage'],
          marginal_cost     = tech_df['marginal cost']['storage']
          )

# ----- Add hydrogen production --------------------
#Add "loads" in the form of negative generators
if add_hydrogen:
    n.add("Generator",
          "P2X",
          bus               = bus_df.loc['Energy Island']['Bus name'], # Add to island bus
          carrier           = "P2X",
          p_nom_extendable  = True,
          p_max_pu          = 0,
          p_min_pu          = -1,
          capital_cost      = tech_df['capital cost']['hydrogen'],
           marginal_cost    = tech_df['marginal cost']['hydrogen'],
          )

# ----- Add datacenter --------------------
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

if add_moneybin:
    n.add("Generator",
          "MoneyBin",
          bus               = bus_df.loc['Energy Island']['Bus name'], # Add to island bus
          carrier           = "MoneyBin",
          p_nom_extendable  = True,
          p_nom_min         = 1,
          p_nom_max         = 1,
          capital_cost      = island_area/n.area_use['data']*tech_df['marginal cost']['datacenter']*8760,
          marginal_cost     = island_area/n.area_use['data']*tech_df['marginal cost']['datacenter'],
          )

# %% Extra functionality
def extra_functionalities(n, snapshots):
    gm.area_constraint(n, snapshots)
    gm.link_constraint(n, snapshots)     

#%% Solver
n.lopf(pyomo = False,
             solver_name = 'gurobi',
              extra_functionality = extra_functionalities,
             )

#%% Save network

filename = '/case2_recreate.nc'
export_path = os.getcwd() + filename
n.export_to_netcdf(export_path)


#%%
gm.its_britney_bitch()