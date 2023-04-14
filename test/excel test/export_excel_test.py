# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 11:38:59 2023

@author: lukas
"""

import os
import sys
# Add modules folder to path
sys.path.append(os.path.abspath('../../modules')) 

import gorm as gm
import tim as tm

import pandas as pd

#%% Input

year = 2030

connected_countries =  [
                        "Denmark",         
                        "Norway",          
                        "Germany",         
                        "Netherlands",     
                        "Belgium",         
                        "United Kingdom"
                        ]

n_std = 1

r = 0.07

n_hrs = 8760

#%% Get all dataframes

# Main parameters ---------------------------------------------------------
main_d = pd.DataFrame(['Main parameters for the island by year',
                       'Source:   Danish Ministry of Climate, Energy and Utilities   -   PDF "ENERGI Ø - FAKTAARK", https://web.archive.org/web/20210206015602/https://kefm.dk/Media/4/A/faktaark%20om%20Energi%C3%B8.pdf'])
main_parameters = tm.get_main_parameters()
main_parameters.index = ['Wind capacity [MW]', 'Island area [m^2]']

# Prices and loads ---------------------------------------------------------
price_d = pd.DataFrame(['Time series of country electricity prices in [Euro/MW], extracted form PyPSA-Eur-Sec-30-Path'])
load_d  = pd.DataFrame(['Time series of country loads in [MW], extracted from PyPSA-Eur-Sec-30-Path to account for electrification'])

price_30_t = pd.DataFrame(['For 2030'])
price_40_t = pd.DataFrame(['For 2040'])

load_30_t = pd.DataFrame(['For 2030'])
load_40_t = pd.DataFrame(['For 2040'])

price_30, load_30 = tm.get_load_and_price(year = 2030, 
                                       connected_countries = connected_countries,
                                       n_std = n_std)

price_40, load_40 = tm.get_load_and_price(year = 2040, 
                                       connected_countries = connected_countries,
                                       n_std = n_std)

# Bus df ---------------------------------------------------------
bus_df_d = pd.DataFrame(['Location that each bus represents along with abbreviation and longitude and Latitude'])
bus_df = tm.get_bus_df(connected_countries = connected_countries)

# Area use data ---------------------------------------------------------
area_use_d = pd.DataFrame(['Area use coefficient for each technology on the island'])

area_use = tm.get_area_use()
area_use.name = 'Area use, [m^2/MW]'

area_use_df = pd.DataFrame(area_use)

area_use_df['source'] = ['Based on EVESCO containerized lithium battery - https://www.power-sonic.com/battery-energy-storage-systems/',
                         'Based on VERDE-20MW containerized alkaline electrolyser, https://verdehydrogen.com/verde-100-electrolyser.html',
                         'Based on the average of two containerized solutions, [1] Kstar IDB containerized data center, https://www.kstar.com/OutdoorDC/17762.jhtml, and [2] Rittal Modular data centres, https://www.rittal.com/imf/none/5_4294/Rittal_Modular_data_centres_in_containers_5_4294/',
                         ]


# Tech Data ---------------------------------------------------------
tech_d = pd.DataFrame(['Capital and Marginal cost for technologies. Some technologies provide a gain for the system, modeled as marginal revenue.',
                       'All numbers are for a discount rate of ' + str(r) + '.'])

tech_30_data = tm.get_tech_data(year = 2030, r = r, n_hrs = n_hrs)
tech_30_data.columns = ['capital cost (CC) [Euro/MW]', 'marginal cost (CC) \n [Euro/MWh]']

# Tech sources
tech_30_data['CC source'] = ['✓  Danish Energy Agency - Technology Data for Generation of Electricity and District Heating',
                         '✓  Danish Energy Agency - Technology Data for Renewable Fuels',
                         '✓  Danish Energy Agency - Technology Data for Energy Storage',
                         '✓  Calculated cost - hardware (https://www.thinkmate.com/system/gigabyte-h273-z82-(rev.-aaw1)), plus containerization assumptions.',
                         '✓  PyPSA Technology Data: https://github.com/PyPSA/technology-data/blob/master/inputs/manual_input.csv',
                         ]

tech_30_data['MC source'] = ['✓  Danish Energy Agency - Technology Data for Generation of Electricity and District Heating',
                         '✓  Lazard LCOH analysis',
                         '✓  Danish Energy Agency - Technology Data for Energy Storage',
                         '✓  https://genome.au.dk/ DKK/CPUhr converted to DKK/MW ',
                         'X  Assumed',
                         ]

tech_30_data['notes'] = [ ' ', 
                      'Marginal revenue', 
                      ' ', 
                      'Marginal revenue', 
                      'Link capital cost is in [Euro/MW/km]'
                      ]

tech_40_data = tm.get_tech_data(year = 2040, r = r, n_hrs = n_hrs)
tech_40_data.columns = ['capital cost [Euro/MW]', 'marginal cost [Euro/MWh]']

# Tech sources
tech_40_data['CC source'] = tech_30_data['CC source']
tech_40_data['MC source'] = tech_30_data['MC source']

tech_40_data['notes'] = tech_30_data['notes']

# Front dataframe ---------------------------------------------------------
description = ' Landing page, short intro to what this file contains.'

front = pd.DataFrame(
     [[description, None],
      [None, None],
      ['Parameters used for calculation:', None],
      ['Discount rate', r],
      ['n_hrs', n_hrs],
      ['n_std', n_std],
      ],
    )

#%% Write to sheets

writer = pd.ExcelWriter('model_data.xlsx', engine = 'xlsxwriter')

# Front ---------------------------------------------------------
front.to_excel(writer, sheet_name = 'description',
               index = False, header = False)

# Main parameters ---------------------------------------------------------
main_d.to_excel(writer, sheet_name = 'main parameters',
                              index = False, header = False)
main_parameters.to_excel(writer, sheet_name = 'main parameters', 
                         startrow = 3, startcol = 1)

# Prices ---------------------------------------------------------
price_d.to_excel(writer, sheet_name = 'prices',
                              index = False, header = False)
price_30.to_excel(writer, sheet_name = 'prices', startrow = 3, startcol = 1)
price_40.to_excel(writer, sheet_name = 'prices', startrow = 3, startcol = 9)

price_30_t.to_excel(writer, sheet_name = 'prices', startrow = 2, startcol = 1,
                    index = False, header = False)
price_40_t.to_excel(writer, sheet_name = 'prices', startrow = 2, startcol = 9,
                    index = False, header = False)

# Loads ---------------------------------------------------------
load_d.to_excel(writer, sheet_name = 'loads',
                              index = False, header = False)
load_30.to_excel(writer, sheet_name = 'loads', startrow = 3, startcol = 1,)
load_40.to_excel(writer, sheet_name = 'loads', startrow = 3, startcol = 9)

load_30_t.to_excel(writer, sheet_name = 'loads', startrow = 2, startcol = 1,
                   index = False, header = False)
load_40_t.to_excel(writer, sheet_name = 'loads', startrow = 2, startcol = 9,
                   index = False, header = False)

# Bus df ---------------------------------------------------------
bus_df_d.to_excel(writer, sheet_name = 'buses',
                              index = False, header = False)
bus_df.to_excel(writer, sheet_name = 'buses', startrow = 2, startcol = 1)

# Area ---------------------------------------------------------
area_use_d.to_excel(writer, sheet_name = 'area use',
                              index = False, header = False)
area_use_df.to_excel(writer, sheet_name = 'area use', startrow = 2, startcol = 1)

# Tech ---------------------------------------------------------
tech_d.to_excel(writer, sheet_name = 'tech_data',
                              index = False, header = False)
tech_30_data.to_excel(writer, sheet_name = 'tech_data',
                   startrow = 3, startcol = 1)

tech_40_data.to_excel(writer, sheet_name = 'tech_data',
                   startrow = 10, startcol = 1)

# Close writer---------------------------------------------------------
writer.close()



















