# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 09:29:02 2023

@author: lukas
"""

import pypsa
from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints, get_dual, get_con, write_objective, get_sol, define_variables
import pandas as pd
import gorm as gm

import matplotlib.pyplot as plt
import island_plt as ip
ip.set_plot_options()

import os
import sys

#%% ------- CONTROL -----------------------------------

should_plot = True

year     = 2030        # Choose year
n_hrs    = 8760        # [hrs] Choose number of hours to simulate
island_area = 120_000  # [m^2] total island area
link_efficiency = 0.97 

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
country_df      = bus_df[1:].reset_index(drop = True).copy() 

# ----- Dataframe with tech data ---------
tech_df         = gm.get_tech_data()

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
for country in country_df['Bus name'].values:
    gm.add_bi_link(n,
                   bus0          = bus_df['Bus name'].values[0],   # From Energy island
                   bus1          = country,                # To country bus
                   link_name     = "Island to " + country, # Name link
                   efficiency    = link_efficiency,
                   capital_cost  = tech_df['marginal cost']['link'],
                   marginal_cost = tech_df['capital cost']['link'],
                   carrier       = 'DC',
                   p_nom_extendable = True,
                   )

### COUNTRIES ###
# ----- Add generators for countries--------------------
#Add generators to each country bus with varying marginal costs
for i in range(0, country_df.shape[0]):
    n.add(
        "Generator",
        name             = "Gen " + country_df['Bus name'][i],
        bus              = country_df['Bus name'][i],
        capital_cost     = 0,            
        marginal_cost    = cprice[country_df['Abbreviation'][i]].values,
        p_nom_extendable = True,   #Make sure country can always deliver to price
        )
    
# ----- Add loads for countries--------------------
# Add loads to each country bus
for i in range(0, country_df.shape[0]): #i becomes integers
    n.add(
        "Load",
        name    = "Load " + country_df['Bus name'][i],
        bus     = country_df['Bus name'][i],
        p_set   = cload[country_df['Abbreviation'][i]].values,
        )   

### ISLAND ###
# ----- Add wind generator --------------------
n.add("Generator",
      "Wind",
      bus               = "Island",
      carrier           = "wind",
      p_nom_extendable  = True,
      p_nom_min         = 3_000,
      p_nom_ma          = 3_000,
      p_max_pu          = wind_cf['electricity'].values,
      marginal_cost     = tech_df['marginal cost']['wind turbine'],
      capital_cost      = tech_df['capital cost']['wind turbine']
      )


#%% Plot

if should_plot:
    ip.plot_geomap(n)


































