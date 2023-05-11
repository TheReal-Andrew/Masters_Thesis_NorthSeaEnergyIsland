# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:46:14 2023

@author: lukas
"""
import pypsa
import numpy as np
import gorm as gm

year = 2040
project_name = 'local_nac'

solutions   = np.load(f'v_{year}_{project_name}_3MAA_10p_solutions.npy')
n_opt       = pypsa.Network(f'v_{year}_{project_name}_opt.nc')
techs       = ['P2X', 'Data', 'Store1']
techs       = techs[:2]
#%%

sweeps = gm.sweep_solutions(n_opt, solutions[:,[0,1]], techs = techs,
                            sweep_range = 2)


