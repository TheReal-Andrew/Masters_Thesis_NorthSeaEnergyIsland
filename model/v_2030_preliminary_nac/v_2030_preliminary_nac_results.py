# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 13:19:05 2023

@author: lukas
"""

#%% Import

import os
import sys
# Add modules folder to path
os.chdir(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.abspath('../../modules')) 

import pypsa
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

import gorm as gm
import tim as tm
from ttictoc import tic,toc
gm.set_plot_options()

#%% Results

solutions = np.load('v_2030_preliminary_nac_3MAA_10p_solutions.npy')

techs = ['P2X', 'Data', 'Store1']

gm.solutions_2D(techs, solutions, n_samples = 1000,
                alpha = 1,
                title = '2D plot of 3D MAA space, without area constraint',
                filename = 'v_2030_preliminary_nac_3MAA_10p_2D_MAA_plot.pdf')


