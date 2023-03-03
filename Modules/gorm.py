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
    
#%% ------- CALCULATION / OPERATION FUNCTIONS ---------------------------------

# Get annuity Annuity from discount rate (i) and lifetime (n)
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

def get_earth_distance(lat1,lat2,lon1,lon2):
    import numpy as np
    R = 6378.1  #Earths radius
    dlon = (lon2 - lon1) * np.pi/180
    dlat = (lat2 - lat1) * np.pi/180
    
    lat1 = lat1 * np.pi/180
    lat2 = lat2 * np.pi/180
    
    a = (np.sin(dlat/2))**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2))**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)) 
    d = R*c #Spherical distance between two points
    
    return d
    
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
        [                                #Create numpy array with bus info
        ["Energy Island",   "EI",   6.68, 56.52], #Energy Island
        ["Denmark",         "DK",   8.12, 56.37], #Assumed Thorsminde
        ["Norway",          "NO",   8.02, 58.16], #Assumed Kristiansand
        ["Germany",         "DE",   8.58, 53.54], #Assumed Bremerhaven
        ["Netherlands",     "NL",   6.84, 53.44], #Assumed Eemshaven
        ["Belgium",         "BE",   3.20, 51.32], #Assumed Zeebrugge
        ["United Kingdom",  "GB",  -1.45, 55.01], #Assumed North Shield
        ], 
        columns = ["Bus name", "Abbreviation", "X", "Y"]   #Give columns titles
        )
    
    # Return only the included countries
    connections = "Island"
    
    for country in connected_countries:
        connections = connections + "|" + country
        
    bus_df = (bus_df[bus_df['Bus name']
                     .str.contains(connections)] # Include only selected countries
                     .reset_index(drop = True)   # Reset index
                     .set_index('Bus name', drop = False)
                     )
    
    return bus_df

def get_area_use():
    # Get area use for technologies, in [m^2/MW] or [m^2/MWh]. Estimated from
    # containerized versions of these technologies.
    import pandas as pd
    
    area_use = pd.Series( data = {'storage':0.9,  #[m^2/MWh] Capacity
                                  'hydrogen':12,  #3.7, #[m^2/MW] capacity
                                  'data':33,      #[m^2/MW] IT output
                                  })
    
    return area_use

def get_tech_data(year = 2030, r = 0.07):
    # Create dataframe with technology data from year and discount rate (r)
    import pandas as pd 
    import numpy as np
    
    DR = 1.2 # Detour factor for links
    
    # ------------------- Raw Data Import --------------------------
    # ----- Pypsa Technology Data -----
    # Load data from pypsa tech data, and set multiindex.
    # Multiindex is technology -> parameter -> year
    pypsa_data = pd.read_csv('https://github.com/PyPSA/technology-data/blob/master/inputs/manual_input.csv?raw=true',
                             index_col = [0, 1, 2])
    
    # Extract HVDC submarine link data values and unstack
    link_data = pypsa_data['value'].loc['HVDC submarine'].unstack(level = 0)
    
    # ----- Battery Storage Data -----
    storage_data = (pd.read_excel(r'../../data/costs/technology_data_catalogue_for_energy_storage.xlsx',
                                 sheet_name = '180 Lithium Ion Battery',
                                 skiprows = [0, 1, 2, 3], # Drop initial empty columns
                                 usecols = 'B:K') # Use specific columns
                    .dropna(axis = 1, how = 'all').dropna(axis = 0, how = 'all') # Remove empty rows/columns
                    .set_index('Technology') # set index
                    )

    storage_data.columns = storage_data.iloc[0,:] # Change column names
    storage_data = storage_data[4:35]             # Get only relevant data
    storage_data = storage_data[year]
    
    # ----- Hydrogen -----
    hydrogen_data = (pd.read_excel(r'../../data/costs/data_sheets_for_renewable_fuels.xlsx',
                                 sheet_name = '86 PEMEC 100MW',
                                 skiprows = [0, 1], # Drop initial empty columns
                                 usecols = 'B:F') # Use specific columns
                    .dropna(axis = 1, how = 'all').dropna(axis = 0, how = 'all') # Remove empty rows/columns
                    .set_index('Technology') # set index
                    )

    hydrogen_data.columns = hydrogen_data.iloc[0,:] # Change column names
    hydrogen_data = hydrogen_data[2:29]             # Get only relevant data
    hydrogen_data = hydrogen_data[year]
    
    # ----- Wind -----
    wind_data = (pd.read_excel(r'../../data/costs/technology_data_for_el_and_dh.xlsx',
                                 sheet_name = '21 Offshore turbines',
                                 skiprows = [0, 2], # Drop initial empty columns
                                 usecols = 'A:G',
                                 index_col = [0, 1]
                                 ) # Use specific columns
                    .dropna(axis = 1, how = 'all').dropna(axis = 0, how = 'all') # Remove empty rows/columns
                    )

    wind_data = wind_data[year]
    
    # ------------------- Cost calculations --------------------------
    # ----- wind -----
    cc_wind       = (get_annuity(r, wind_data['Energy/technical data']['Technical lifetime [years]'])
                     * wind_data['Financial data']['Nominal investment (*total) [2020-MEUR/MW_e]'] * 1e6
                     + wind_data['Financial data']['Fixed O&M (*total) [2020-EUR/MW_e/y]']
                     )# [euro/MW] From Energistyrelsen
    
    mc_wind       = wind_data['Financial data']['Variable O&M (*total) [2020-EUR/MWh_e]']   # [euro/MWh] From Energistyrelsen
    
    # ----- hydrogen -----
    cc_hydrogen   = (get_annuity(r, hydrogen_data['Technical lifetime (years)'])
                     * hydrogen_data['Specific investment (€ / kW of total input_e)'] * 1e3
                     * (1+ (hydrogen_data['Fixed O&M (% of specific investment / year) ']*0.01))
                     ) # [euro/MW]  From Energistyrelsen
    
    mc_hydrogen   = 127 # [euro/MWh] Revenue - UNCONFIRMED
    
    # ----- storage -----
    cc_storage    = (get_annuity(r,storage_data['Technical lifetime (years)']) # Annuity
                          * storage_data['Specific investment (M€2015 per MWh)'] * 1e6 # Investment
                          + storage_data['Fixed O&M (k€2015/MW/year)'] * 1e3 # O&M
                          ) # [euro/MWh] From Energistyrelsen
    mc_storage    = (storage_data['Variable O&M (€2015/MWh)']) # [euro/MWh]  From Energistyrelsen
    
    # ----- datacenter -----
    cc_datacenter = 1e6 # [euro/MW]  UNCONFIRMED
    mc_datacenter = 120    # [euro/MWh] revenue UNCONFIRMED
    
    # ----- link -----
    # Link, based on pypsa tech data. cc returns capital cost per km!
    cc_link       = float( get_annuity(r, link_data['lifetime']) # [euro/MW/km]  
                                 * link_data['investment']
                                 * DR
                                 ) 
    mc_link       = 5 #38 [euro/MWh] UNCONFIRMED
    
    # ---- Assemble tech dataframe
    tech_df = pd.DataFrame(
        [                             
        ["wind turbine",  cc_wind,        mc_wind      ], 
        ["hydrogen",      cc_hydrogen,    mc_hydrogen  ], 
        ["storage",       cc_storage,     mc_storage   ], 
        ["datacenter",    cc_datacenter,  mc_datacenter], 
        ["link",          cc_link,        mc_link      ], 
        ], 
        columns = ["tech", "capital cost", "marginal cost"]   #Give columns titles
        )
    
    tech_df = tech_df.set_index('tech')
    
    return tech_df

#%% ------- PyPSA FUNCTIONS -----------------------------------

# Add bidirectional link with setup for losses
def add_bi_link(network, bus0, bus1, link_name, carrier, efficiency = 1,
               capital_cost = 0, marginal_cost = 0, 
               p_nom_extendable = True, p_nom_max = float('inf'),
               p_nom_min = 0,
               x = None, y = None, bus_shift = [0, 0]):
    # Function that adds a bidirectional link with efficiency and marginal cost
    # between two buses. This is done by adding additional "efficiency buses" 
    # (e0 and e1) for each bus, and running a bidirectional lossless link 
    # between them. The unidirectional links from e0 and e1 to their respective
    # buses then take care of adding correct marignal cost and efficiency.
    # capital cost is added on the bidirectional lossless link, while effieincy
    # and maringla cost are added on the links from efficiency buses to buses.
    
    # ---- Add efficiency buses at each end ----
    e0 = link_name + '_e0'
    e1 = link_name + '_e1'
    
    network.madd('Bus',
          names = [e0,  e1],
          x     = [network.buses.loc[network.buses.index == bus0]['x'] + bus_shift[0] , 
                   network.buses.loc[network.buses.index == bus1]['x'] + bus_shift[0]],
          y     = [network.buses.loc[network.buses.index == bus0]['y'] + bus_shift[1], 
                   network.buses.loc[network.buses.index == bus1]['y'] + bus_shift[1]],
          )
    
    # ---- main bidirectional Link ----
    # capital_cost and marginal_cost are applied here
    network.add('Link',
          link_name,
          bus0              = e0,
          bus1              = e1,
          p_min_pu          = -1,
          p_nom_extendable  = p_nom_extendable,
          capital_cost      = capital_cost,    #Capital cost is added here
          p_nom_max         = p_nom_max,
          p_nom_min         = p_nom_min,
          carrier           = carrier,
          )
    
    # ---- Links on bus 0 ----
    # link from buses to ebuses
    network.madd('Link',
          names         = [link_name + '_bus0_to_e0',  link_name + '_e0_to_bus0'],
          bus0          = [bus0,                       e0          ],
          bus1          = [e0,                         bus0        ],
          efficiency    = [1,                          efficiency  ],
          marginal_cost = [0,                          marginal_cost],
          p_nom_extendable  = [True, True],
          )
    
    # ---- Links on bus 0 ----
    # link from buses to ebuses
    network.madd('Link',
          names         = [link_name + '_bus1_to_e1',  link_name + '_e1_to_bus1'],
          bus0          = [bus1,                          e1          ],
          bus1          = [e1,                            bus1        ],
          efficiency    = [1,                             efficiency  ],
          marginal_cost = [0,                             marginal_cost],
          p_nom_extendable  = [True, True],
          )