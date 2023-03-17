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
import pypsa_diagrams as pdiag

import matplotlib.pyplot as plt
import island_plt as ip
ip.set_plot_options()

#%% ------- CONTROL -----------------------------------

# Main control
should_solve       = True
should_plot        = True
should_bus_diagram = False
should_n_diagram   = True

# Main parameters
year     = 2030        # Choose year
r        = 0.07        # Discount rate
wind_cap = 3000       # [MW] Installed wind capacity
n_hrs    = 8760        # [hrs] Choose number of hours to simulate
island_area = 120_000  # [m^2] total island area

link_efficiency = 0.95          # Efficiency of links
link_total_max  = wind_cap      # Total allowed link capacity
link_p_nom_min  = 0           # Minimum allowed capacity for one link
link_limit      = float('inf')  # [MW] Limit links to countries. float('inf')

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
add_c_gens  = True # Add country generators
add_c_loads = True # Add country demand

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

# Add data to network for easier access when creating constraints
n.bus_df         = bus_df
n.area_use       = area_use
n.total_area     = island_area 
n.link_total_max = link_total_max
n.link_p_nom_min = link_p_nom_min

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
                    p_nom_max     = link_limit, # [MW]
                    p_nom_min     = link_p_nom_min,
                    bus_shift     = jiggle,
                    )
    
# Add list of main links to network to differetniate
n.main_links = n.links[~n.links.index.str.contains("bus")].index

### COUNTRIES ###
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
      # capital_cost      = tech_df['capital cost']['wind turbine'],
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
            marginal_cost     = tech_df['marginal cost']['datacenter'] ,
            )

#%% Extra functionality
def area_constraint(n, snapshots):
    
    # Get variables for all generators and store
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
    # Get main links 
    link_names = n.main_links               # List of main link names
    link_t     = n.link_total_max           # Maximum total link capacity
    link_min   = n.link_p_nom_min           # Minimum link capacity
    link_t_min = link_min * len(link_names) # Minimum total link capacity
    
    # get all link variables, and then get only main link variables
    vars_links   = get_var(n, 'Link', 'p_nom')
    vars_links   = vars_links[link_names]
    
    # Sum up link capacities of chosen links (lhs), and set limit (rhs)
    rhs          = link_t if link_t >= link_t_min else link_t_min
    lhs          = join_exprs(linexpr((1, vars_links)))
    
    #Define constraint and name it 'Total constraint'
    define_constraints(n, lhs, '=', rhs, 'Link', 'Total constraint')

def extra_functionalities(n, snapshots):
    area_constraint(n, snapshots)
    link_constraint(n, snapshots)

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

ip.set_plot_options()

if should_plot:
    ip.plot_geomap(n)

if should_n_diagram:
    
    pos = [
           [0, 0], #Island
           [20, -1], #Denmark
           [15, 8],  #Norway
           [18, -10], #DE
           [6, -11], #NE
           [-4, -12], #BE
           [-10, 1], #UK
          ]
    
    pdiag.draw_network(n, spacing = 1, handle_bi = True, pos = None,
                       bus_color = 'azure',
                       filename = 'pypsa_diagram_3_2.svg')
    
if should_bus_diagram:
    pdiag.draw_bus(n, 'Energy Island', bus_color = 'azure',
                   handle_bi = True, link_line_length = 1.1,
                   filename = 'bus_diagram1.svg')
    
    



# t2 = pd.date_range('2030-01-01 00:00', '2030-01-07 00:00', freq = 'H')
# ax = n.generators_t.p['Data'][t2].abs().plot(figsize = (15,5))
# fig = plt.gcf()
# ax.set_xlabel('Time [hr]')
# ax.set_ylabel('Power consumed [MW]')
# ax.set_title('Data')
# fig.savefig('Data_timeseries.svg', format = 'svg', bbox_inches='tight')


# Extra
# linkz = n.links_t.p0

# linkz2 = n.links[~n.links.index.str.contains("bus")]

# country = 'Denmark'
# linkz_country = n.links[n.links.index.str.contains(country)]

# linkz2['p_nom_opt'].hist(bins = 12, figsize = (10,5))
# linkz2['p_nom_opt'].plot(figsize = (10,5))






























