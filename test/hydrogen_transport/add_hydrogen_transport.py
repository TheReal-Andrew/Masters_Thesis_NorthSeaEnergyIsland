# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 19:05:38 2023

@author: lukas
"""

import pandas as pd
import gorm as gm

year = 2030

hydrogen_transport = (pd.read_excel(r'../../data/costs/energy_transport_datasheet.xlsx',
                      sheet_name = 'H2 140',
                      skiprows = [0, 1], # Drop initial empty columns
                      usecols = 'B:E')
                      .dropna(axis = 1, how = 'all').dropna(axis = 0, how = 'all')
                      .set_index('Technology') # set index
                      )
         
hydrogen_transport = hydrogen_transport[year]   

cost = (gm.get_annuity(0.07, hydrogen_transport['Technical life time (years)']) 
        * hydrogen_transport['Investment costs; single line, 1000-4000 MW (EUR/MW/m)']*1000*100 
        +  (hydrogen_transport['Fixed O&M (EUR/km/year/MW)']*100)
        ) # [Eur/MW]
    


          
