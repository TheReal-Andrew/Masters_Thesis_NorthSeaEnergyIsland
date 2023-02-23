# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 20:21:54 2023

@author: lukas
"""
#%%
# General cOde Resource Module (GORM) - module containing useful functions

#%% -------GENERAL FUNCTIONS -----------------------------------

# Play sound from folder
def its_britney_bitch(path):
    import os
    import random 

    files_in_path   = os.listdir(path) # Get files in directory
    file_to_play    = random.choice(files_in_path) # Choose random file to play
    
    # Play file
    os.startfile(path + '/' + file_to_play)
    
#%% ------- DATA FUNCTIONS -----------------------------------
def get_load_and_price(year, connected_countries, n_std): # require year
    import pandas as pd
    
    # load data
    cprice = pd.read_csv('https://raw.githubusercontent.com/TheReal-Andrew/Pre_project/main/Data/market/price_%d.csv'%year, index_col = 0)      
    cload = pd.read_csv('https://raw.githubusercontent.com/TheReal-Andrew/Pre_project/main/Data/market/load_%d.csv'%year, index_col = 0)

    # Get bus_df with selected country abbreviations
    bus_df      = get_bus_df(connected_countries)

    # Get only data for the selected countries
    cprice = cprice[bus_df['Abbreviation'].values[1:]]
    cload  = cload[bus_df['Abbreviation'].values[1:]]
    
    # Remove outliers
    cprice = remove_outliers(cprice, cprice.keys(), n_std)
    
    return cprice, cload

def get_bus_df(connected_countries):
    import pandas as pd
    import numpy as np
    
    #Create dataframe with info on buses. Names, x (Longitude) and y (Latitude) 
    bus_df = pd.DataFrame(
        np.array([                                #Create numpy array with bus info
        ["Energy Island",   "EI",   6.68, 56.52], #Energy Island
        ["Denmark",         "DK",   8.12, 56.37], #Assumed Thorsminde
        ["Norway",          "NO",   8.02, 58.16], #Assumed Kristiansand
        ["Germany",         "DE",   8.58, 53.54], #Assumed Bremerhaven
        ["Netherlands",     "NL",   6.84, 53.44], #Assumed Eemshaven
        ["Belgium",         "BE",   3.20, 51.32], #Assumed Zeebrugge
        ["United Kingdom",  "GB",  -1.45, 55.01], #Assumed North Shield
        ]), 
        columns = ["Bus name", "Abbreviation", "X", "Y"]   #Give columns titles
        )
    
    # Return only the included countries
    connections = "Island"
    
    for country in connected_countries:
        connections = connections + "|" + country
        
    bus_df = (bus_df[bus_df['Bus name']
                     .str.contains(connections)] # Include only selected countries
                     .reset_index(drop = True)   # Reset index
                     )
    
    return bus_df

def get_area_use():
    # Get area use for technologies, in [m^2/MW] or [m^2/MWh]. Estimated from
    # containerized versions of these technologies.
    import pandas as pd
    
    area_use = pd.Series( data = {'storage':0.9,  #[m^2/MWh] Capacity
                                  'hydrogen':3.7, #[m^2/MW] capacity
                                  'data':30,      #[m^2/MW] IT output
                                  })
    
    # Insert into network for easier import in extra_functionalities
    # n.area_use = area_use
    
    return area_use

def get_tech_data():
    # Create dataframe with technology data
    import pandas as pd 
    import numpy as np
    
    # ---- Cost calculations
    # wind
    cc_wind       = get_annuity(0.07, 30) * 1.8e6 # [euro/MW]
    mc_wind       = 0                             # [euro/MWh] 
    
    # hydrogen
    cc_hydrogen   = 1 # [euro/MW] 
    mc_hydrogen   = 1 # [euro/MWh] 
    
    # storage
    cc_storage    = 1 # [euro/MW]
    mc_storage    = 1 # [euro/MWh] 
    
    cc_datacenter = 1 # [euro/MW]
    mc_datacenter = 1 # [euro/MWh] 
    
    cc_link       = 1 # [euro/MW]
    mc_link       = 1 # [euro/MWh] 
    
    # ---- Assemble tech dataframe
    tech_df = pd.DataFrame(
        np.array([                             
        ["wind turbine",  cc_wind,        mc_wind      ], 
        ["hydrogen",      cc_hydrogen,    mc_hydrogen  ], 
        ["storage",       cc_storage,     mc_storage   ], 
        ["datacenter",    cc_datacenter,  mc_datacenter], 
        ["link",          cc_link,        mc_link      ], 
        ]), 
        columns = ["tech", "capital cost", "marginal cost"]   #Give columns titles
        )
    
    tech_df = tech_df.set_index('tech')
    
    return tech_df

#%% ------- CALCULATION / OPERATION FUNCTIONS ---------------------------------

# Get annuity Annuity from discount rate and lifetime
def get_annuity(i, n):
    annuity = i/(1.-1./(1.+i)**n)
    return annuity

# Remove outliers from dataframe, and replace with n standard deviations.
def remove_outliers(df,columns,n_std):
    for col in columns:
        print('Removing outliers - Working on column: {}'.format(col))
        
        df[col][ df[col] >= df[col].mean() + (n_std * df[col].std())] = \
        df[col].mean() + (n_std * df[col].std())
        
    return df

#%% ------- PyPSA FUNCTIONS -----------------------------------

# Add bidirectional link with setup for losses
def add_bi_link(network, bus0, bus1, link_name, carrier, efficiency = 1,
               capital_cost = 0, marginal_cost = 0, p_nom_extendable = True,
               x = None, y = None):
    
    # ---- Add efficiency buses at each end ----
    network.madd('Bus',
          names = [link_name + '_ebus0',  link_name + '_ebus1',],
          x     = [network.buses.loc[network.buses.index == bus0]['x'], 
                   network.buses.loc[network.buses.index == bus1]['x']],
          y     = [network.buses.loc[network.buses.index == bus0]['y'], 
                   network.buses.loc[network.buses.index == bus1]['y']],
          )
    
    # ---- main bidirectional Link ----
    # capital_cost and marginal_cost are applied here
    network.add('Link',
          link_name,
          bus0              = link_name + '_ebus0',
          bus1              = link_name + '_ebus1',
          p_min_pu          = -1,
          p_nom_extendable  = p_nom_extendable,
          capital_cost      = capital_cost,
          marginal_cost     = marginal_cost,
          carrier           = carrier,
          )
    
    # ---- links from buses to ebuses ----
    # link from buses to ebuses
    network.madd('Link',
          names       = [link_name + '_ebus0_elink0',  link_name + '_ebus1_elink0'],
          bus0        = [bus0, bus1],
          bus1        = [link_name + '_ebus0', link_name + '_ebus1'],
          carrier     = carrier,
          p_nom_extendable  = True,
          )
    
    #links from ebuses to buses (Efficiency applied here)
    network.madd('Link',
          [link_name + '_ebus0_elink1',  link_name + '_ebus1_elink1'],
          bus0        = [link_name + '_ebus0',  link_name + '_ebus1'],
          bus1        = [bus0, bus1],
          efficiency  = efficiency,
          carrier     = carrier,
          p_nom_extendable  = True,
          )