# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 12:08:53 2023

@author: lukas
"""

import pypsa
import pandas as pd
import numpy as np
import os
import sys
# Add modules folder to path
sys.path.append(os.path.abspath('../../modules')) 

import gorm as gm
import island_plt as ip

n = pypsa.Network()
t = pd.date_range('2030-01-01 00:00', '2030-12-31 23:00', freq = 'H')
n.set_snapshots(t)

wind_cf         = pd.read_csv(r'data\wind_formatted.csv',
                       index_col = [0], sep=",")

tech_df         = gm.get_tech_data(2030, 0.07)

cprice, cload   = gm.get_load_and_price(2030, ["Denmark"], n_std = 1)

n.madd('Bus',
       names = ['Island', 'DK'],
       x = [6,  8],
       y = [56, 57])

n.add(
    "Generator",
    name             = "DK Gen",
    bus              = "DK",
    capital_cost     = 0,            
    marginal_cost    = cprice['DK'].values,
    p_nom_extendable = True,   #Make sure country can always deliver to price
    )


n.add("Generator",
      "Wind",
      bus               = 'Island', # Add to island bus
      carrier           = "wind",
      p_nom_extendable  = True,
      p_nom_min         = 3000,
      p_nom_max         = 3000,
      p_max_pu          = wind_cf['electricity'].values,
      capital_cost      = tech_df['capital cost']['wind turbine'],
      marginal_cost     = tech_df['marginal cost']['wind turbine'],
      )

n.add('Store',
      'store1',
      bus = 'Island',
      e_nom_extendable = True,
      capital_cost      = tech_df['capital cost']['storage'],
      marginal_cost     = tech_df['marginal cost']['storage'],
      )

n.add('Load',
      'Load1',
      bus = 'DK',
      p_set = 1000,)

gm.add_bi_link(n, 'DK', 'Island', link_name = 'Link1', carrier = 'DC', efficiency = 0.97)

n.lopf(pyomo = False,
       solver_name = 'gurobi',
       keep_shadowprices = True,
       keep_references = True,
       # extra_functionality = extra_functionalities,
       )

ip.plot_geomap(n)

linkz = n.links_t.p0