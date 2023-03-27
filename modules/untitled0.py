# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 12:33:25 2023

@author: lukas
"""
import os
import pandas as pd
import tim as tm

year = 2030

connected_countries =  [
                        "Denmark",         
                        "Norway",          
                        "Germany",         
                        "Netherlands",     
                        "Belgium",         
                        "United Kingdom"
                        ]

# cload1 = pd.read_csv(f"../data/market/el_demand_adjusted_{year}.csv", index_col = 0)
# cload2 = pd.read_csv('https://raw.githubusercontent.com/TheReal-Andrew/Pre_project/main/Data/market/load_%d.csv'%year, index_col = 0)


cload, cprice = tm.get_load_and_price(2030, connected_countries, 1)

