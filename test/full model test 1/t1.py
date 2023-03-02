# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:20:32 2023

@author: lukas
"""


import pandas as pd

hydrogen_data = (pd.read_excel(r'../../data/costs/data_sheets_for_renewable_fuels.xlsx',
                              sheet_name = '86 PEMEC 100MW',
                              skiprows = [0, 1], # Drop initial empty columns
                              usecols = 'B:F') # Use specific columns
                .dropna(axis = 1, how = 'all').dropna(axis = 0, how = 'all') # Remove empty rows/columns
                .set_index('Technology') # set index
                )

hydrogen_data.columns = hydrogen_data.iloc[0,:] # Change column names
hydrogen_data = hydrogen_data[2:29]             # Get only relevant data
hydrogen_data = hydrogen_data[2030]

wind_data = (pd.read_excel(r'../../data/costs/technology_data_for_el_and_dh.xlsx',
                             sheet_name = '21 Offshore turbines',
                             skiprows = [0, 2], # Drop initial empty columns
                             usecols = 'A:G',
                             index_col = [0, 1]
                             ) # Use specific columns
                .dropna(axis = 1, how = 'all').dropna(axis = 0, how = 'all') # Remove empty rows/columns
                )

wind_data = wind_data[2030]

