# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:00:25 2023

@author: lukas
"""

import pypsa
import numpy as np
import pandas as pd
import island_lib as il
import island_plt as ip
import gorm

ip.set_plot_options()

#%% Network

n = pypsa.Network()

n.madd('Bus',
       ['bus1', 'bus2',  'bus3'],
       )

n.add('Generator',
      'Gen_bus1',
      bus = 'bus2',
      p_nom_extendable = True,
      marginal_cost = 10,
      )

n.add('Load',
      'Load1',
      bus = 'bus1',
      p_set = 100,
       )

n.add('Generator',
      'Gen_bus3',
      bus = 'bus3',
      p_nom_extendable = True,
      p_nom_max = 70,
      marginal_cost = 5,
      )

gorm.add_bilink(n, 'bus1', 'bus2', 'bilink1', efficiency = 0.9)

gorm.add_bilink(n, 'bus2', 'bus3', 'bilink2', efficiency = 0.97)

#%% Efficiency links

# def bi_link(n, bus1, bus2, link_name, efficiency):
    
#     # Bus for efficiency control
#     n.add('Bus',
#           link_name + '_bus',
#           )
    
#     # main bidirectional Link
#     n.add('Link',
#           link_name,
#           bus0              = link_name + '_bus',
#           bus1              = bus2,
#           p_min_pu          = -1,
#           p_nom_extendable  = True,
#           )
    
#     # Additional links to control efficiency
#     n.add('Link',
#           link_name + '_e_1',
#           bus0              = bus1,
#           bus1              = link_name + '_bus',
#           efficiency        = efficiency,
#           p_nom_extendable  = True,
#           )
    
#     n.add('Link',
#           link_name + '_e_2',
#           bus0              = link_name + '_bus',
#           bus1              = bus1,
#           efficiency        = efficiency,
#           p_nom_extendable  = True,
#           )

# bi_link(n, 'bus1', 'bus2', 'bi_link1', 0.9)

# bi_link(n, 'bus2', 'bus3', 'bi_link2', 0.97)


#%% Solve

n.lopf()


















