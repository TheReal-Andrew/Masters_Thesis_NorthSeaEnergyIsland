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

ip.set_plot_options()

#%% Network

n = pypsa.Network()

n.madd('Bus',
       ['bus1', 'bus2'],
       )

n.add('Generator',
      'Gen1',
      bus = 'bus1',
      p_nom_extendable = True,
      )

n.add('Load',
      'Load1',
      bus = 'bus2',
      p_set = 100,
       )

n.add('Link',
      'Link1',
      bus0 = 'bus2',
      bus1 = 'bus1',
      p_min_pu = -1,
      efficiency = 0.9,
      p_nom_extendable = True,
      )

#%% Solve

n.lopf()


















