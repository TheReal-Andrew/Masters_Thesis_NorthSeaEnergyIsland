# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 09:29:02 2023

@author: lukas
"""

import os
import sys
# Add modules folder to path
sys.path.append(os.path.abspath('../../modules')) 

import pypsa
from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints, get_dual, get_con, write_objective, get_sol, define_variables
import pandas as pd

import gorm as gm

import matplotlib.pyplot as plt
import island_plt as ip
ip.set_plot_options()

#%% ------- CONTROL -----------------------------------

# Main control
should_solve = True
should_plot  = True

# Main parameters
year     = 2030        # Choose year
wind_cap = 3000        # [MW] Installed wind capacity
n_hrs    = 8760        # [hrs] Choose number of hours to simulate
island_area = 120_000  # [m^2] total island area
link_efficiency = 0.90 
r        = 0.07        # Discount rate

filename = "network_1_" # Choose filename for export

# Choose which countries to include of this list, comment unwanted out.
connected_countries =  [
                        "Denmark",         
                        "Norway",          
                        "Germany",         
                        "Netherlands",     
                        "Belgium",         
                        "United Kingdom"
                        ]

jiggle = [0, 0]

# Component control
add_storage = True # Add storage on island
add_data    = True # Add datacenter on island
add_hydrogen= True # Add hydrogen production on island

#%% ------- IMPORT DATA -----------------------------------

# ----- Wind capacity factor data ---------
wind_cf         = pd.read_csv(r'data\wind_formatted.csv',
                       index_col = [0], sep=",")[:n_hrs]

# ----- Country demand and price ---------
# Import price and demand for each country for the year, and remove outliers
cprice, cload   = gm.get_load_and_price(year, connected_countries, n_std = 1)

# ----- Dataframe with bus data ---------
# Get dataframe with bus info, only for the connected countries.
bus_df          = gm.get_bus_df(connected_countries) # Import country data for linking
country_df      = bus_df[1:].copy() 

# ----- Dataframe with tech data ---------
tech_df         = gm.get_tech_data(year, r)

# ----- Area use data ---------
area_use        = gm.get_area_use()

#%% ------- NETWORK -----------------------------------------------------------

# ----- initialize network ---------
n = pypsa.Network()
t = pd.date_range('2030-01-01 00:00', '2030-12-31 23:00', freq = 'H')[:n_hrs]
n.set_snapshots(t)

# Add area use and island area to network for easier use in extra_functionalities
n.area_use   = area_use
n.total_area = island_area 

# ----- Add buses-------------------
# Add multiple buses by passing arrays from bus_df to parameters and using madd
n.madd('Bus',
       names = bus_df['Bus name'], 
       x     = bus_df['X'].values,
       y     = bus_df['Y'].values,
        )

# ----- Add links--------------------
for country in country_df['Bus name']:
    
    # Get link distance in [km]
    distance = gm.get_earth_distance(bus_df.loc['Energy Island']['X'],
                                     country_df.loc[country]['X'],
                                     bus_df.loc['Energy Island']['Y'],
                                     country_df.loc[country]['Y'])
    
    # Add bidirectional link with loss and marginal cost
    gm.add_bi_link(n,
                    bus0          = bus_df.loc['Energy Island']['Bus name'],   # From Energy island
                    bus1          = country,                        # To country bus
                    link_name     = "Island to " + country,         # Link name
                    efficiency    = link_efficiency,
                    capital_cost  = tech_df['capital cost']['link'] * distance,
                    marginal_cost = tech_df['marginal cost']['link'],
                    carrier       = 'DC',
                    p_nom_extendable = True,
                    # p_nom_max     = 2000, # [MW]
                    bus_shift     = jiggle,
                    )

### COUNTRIES ###
# ----- Add generators for countries--------------------
#Add generators to each country bus with varying marginal costs
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
for country in country_df['Bus name']:
    n.add('Load',
          name    = "Load " + country,
          bus     = country,
          p_set   = cload[country_df.loc[country]['Abbreviation']].values,
          ) 

### ISLAND ###
# ----- Add wind generator --------------------
n.add("Generator",
      "Wind",
      bus               = bus_df.loc['Energy Island']['Bus name'], # Add to island bus
      carrier           = "wind",
      p_nom_extendable  = True,
      p_nom_min         = wind_cap, # Ensure that capacity is pre-built
      p_nom_max         = wind_cap, # Ensure that capacity is pre-built
      p_max_pu          = wind_cf['electricity'].values,
      capital_cost      = tech_df['capital cost']['wind turbine'],
      marginal_cost     = tech_df['marginal cost']['wind turbine'],
      )

# ----- Add battery storage --------------------
if add_storage:
    n.add("Store",
          "Store",
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
            marginal_cost     = tech_df['marginal cost']['datacenter'] ,
            )

#%% Extra functionality
def area_constraint(n, snapshots):
    vars_gen   = get_var(n, 'Generator', 'p_nom')
    vars_store = get_var(n, 'Store', 'e_nom')
    
    lhs = linexpr((n.area_use['hydrogen'], vars_gen["P2X"]), 
                  (n.area_use['data'],     vars_gen["Data"]), 
                  (n.area_use['storage'],  vars_store))
    
    rhs = n.total_area #[m^2]
    
    define_constraints(n, lhs, '<=', rhs, 'Generator', 'Area_Use')

def extra_functionalities(n, snapshots):
    area_constraint(n, snapshots)

#%% Solve
if should_solve:
    n.lopf(pyomo = False,
           solver_name = 'gurobi',
           keep_shadowprices = True,
           keep_references = True,
            extra_functionality = extra_functionalities,
           )
else:
    pass
    
#%% Plot

if should_plot:
    ip.plot_geomap(n)

linkz = n.links_t.p0

linkz2 = n.links[~n.links.index.str.contains("bus")]

country = 'Denmark'
linkz_country = n.links[n.links.index.str.contains(country)]

linkz2['p_nom_opt'].hist(bins = 12, figsize = (10,5))
linkz2['p_nom_opt'].plot(figsize = (10,5))































