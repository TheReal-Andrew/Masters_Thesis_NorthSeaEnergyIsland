# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 09:30:06 2023

@author: lukas
"""

#%%
# ----- Technology Information Module (TIM) -----
# The TIM module is used to import, process, format and return data to the main
# model script.

#%% ---- MAIN PARAMETERS ----
def get_main_parameters():
    import pandas as pd
    
    mp = pd.DataFrame( {2030 : [3000,  120_000],
                        2040 : [10000, 460_000],
                        2050 : [10000, 460_000],},
                       index = ['wind', 'island_area'])
    
    # Source: https://kefm.dk/Media/4/A/faktaark%20om%20Energi%C3%B8.pdf
    
    return mp

#%% ---- COUNTRY LOAD AND PRICES ----
def get_load_and_price(year = 2030, connected_countries = ['Denmark'], n_std = 1): # require year
    import pandas as pd
    import gorm as gm
    
    # load data
    # cprice = pd.read_csv('https://raw.githubusercontent.com/TheReal-Andrew/Pre_project/main/Data/market/price_%d.csv'%year, index_col = 0)      
    # cload = pd.read_csv('https://raw.githubusercontent.com/TheReal-Andrew/Pre_project/main/Data/market/load_%d.csv'%year, index_col = 0)
    cprice = pd.read_csv(f"../../data/market/price_{year}.csv", index_col = 0)
    
    cload = pd.read_csv(f"../../data/market/el_demand_adjusted_{year}.csv", index_col = 0)
    
    # Get bus_df with selected country abbreviations
    bus_df      = get_bus_df(connected_countries)

    # Get only data for the selected countries
    cprice = cprice[bus_df['Abbreviation'].values[1:]]
    cload  = cload[bus_df['Abbreviation'].values[1:]]
    
    # Remove outliers
    cprice = gm.remove_outliers(cprice, cprice.keys(), n_std)
    
    return cprice, cload

#%% ----- BUS DATAFRAME -----
def get_bus_df(connected_countries = ['Denmark']):
    import pandas as pd
    
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

#%% ----- AREA USE DATA -----
def get_area_use():
    # Get area use for technologies, in [m^2/MW] or [m^2/MWh]. Estimated from
    # containerized versions of these technologies.
    import pandas as pd
    
    area_use = pd.Series( data = {'storage':0.894,  #[m^2/MWh] Capacity
                                  'hydrogen': 2.973,  #[m^2/MW] capacity
                                  'data':18.5,      #[m^2/MW] IT output
                                  })
    
    return area_use

#%% ----- TECH DATA -----
def get_tech_data(year = 2030, r = 0.07, n_hrs = 8760) :
    # Create dataframe with technology data from year and discount rate (r)
    import pandas as pd 
    import gorm as gm
    
    DR = 1.2 # Detour factor for links
    
    # ------------------- Raw Data Import --------------------------
    # ----- Pypsa Technology Data -----
    # Load data from pypsa tech data, and set multiindex.
    # Multiindex is technology -> parameter -> year
    pypsa_data = pd.read_csv('https://github.com/PyPSA/technology-data/blob/master/inputs/manual_input.csv?raw=true',
                             index_col = [0, 1, 2])
    
    # Extract HVDC submarine link data values and unstack
    link_data = pypsa_data['value'].loc['HVDC submarine'].unstack(level = 0)
        # No link data for specific years
    
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
                                 sheet_name = '86 AEC 100MW',
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
    cc_wind       = (gm.get_annuity_snap(r, wind_data['Energy/technical data']['Technical lifetime [years]'], n_hrs)
                     * wind_data['Financial data']['Nominal investment (*total) [2020-MEUR/MW_e]'] * 1e6
                     + wind_data['Financial data']['Fixed O&M (*total) [2020-EUR/MW_e/y]']
                     )# [euro/MW] From Energistyrelsen
    
    mc_wind       = wind_data['Financial data']['Variable O&M (*total) [2020-EUR/MWh_e]']   # [euro/MWh] From Energistyrelsen
    
    # ----- hydrogen -----
    cc_hydrogen   = (gm.get_annuity_snap(r, hydrogen_data['Technical lifetime (years)'], n_hrs)
                     * hydrogen_data['Specific investment (€ / kW of total input_e)'] * 1e3
                     * (1+ (hydrogen_data['Fixed O&M (% of specific investment / year) ']*0.01))
                     ) # [euro/MW]  From Energistyrelsen
    
    mc_hydrogen   = 40.772 # [euro/MWh] Revenue - Lazard LCOE converted using H2 LHV
    
    # ----- storage -----
    cc_storage    = (gm.get_annuity_snap(r, storage_data['Technical lifetime (years)'], n_hrs) # Annuity
                          * storage_data['Specific investment (M€2015 per MWh)'] * 1e6 # Investment
                          + storage_data['Fixed O&M (k€2015/MW/year)'] * 1e3 # O&M
                          ) # [euro/MWh] From Energistyrelsen
    mc_storage    = (storage_data['Variable O&M (€2015/MWh)']) # [euro/MWh]  From Energistyrelsen
    
    # ----- datacenter -----
    cc_datacenter = (gm.get_annuity_snap(r, 5, n_hrs)
                        * (1.88e7)
                        * (1 + 0.02)
                        ) # [euro/MW] Hardware: https://www.thinkmate.com/system/gigabyte-h273-z82-(rev.-aaw1)
    mc_datacenter = 1.118e3  # [euro/MWh] https://genome.au.dk/ gives DKK/CPUhr
    
    # ----- link -----
    # Link, based on pypsa tech data. cc returns capital cost per km!
    cc_link       = float( gm.get_annuity_snap(r, link_data['lifetime'], n_hrs) # [euro/MW/km]  
                                 * link_data['investment']
                                 * DR
                                 ) 
    mc_link       = 0.001 #38 [euro/MWh] 
    
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
