# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:38:14 2023

@author: lukas
"""

import os
import sys
# Add modules folder to path
os.chdir(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.abspath('../modules')) 

import pandas as pd
import gorm as gm
import matplotlib.pyplot as plt
import numpy as np

gm.set_plot_options()

country = 'DK'

data30   = pd.read_csv(f'../data/market/demand_2030.csv', index_col = 0)

# data40   = pd.read_csv(f'../data/market/demand_2040.csv', index_col = 0)

data_a30 = pd.read_csv(f'../data/market/el_demand_adjusted_2030.csv', index_col = 0)

data_a40 = pd.read_csv(f'../data/market/el_demand_adjusted_2040.csv', index_col = 0)

data_a50 = pd.read_csv(f'../data/market/el_demand_adjusted_2050.csv', index_col = 0)

data = pd.concat([data30[country],
                  # data40[country],
                  data_a30[country], 
                  data_a40[country],
                  # data_a50['DK']],
                 ], axis = 1)

data.columns = ['2015 input demand', 
                # '2040 input', 
                '2030 adjusted', 
                '2040 adjusted',
                # 'DK_2050_a',
                ]
data.index = pd.to_datetime(data.index)


#%%
fig, ax = plt.subplots(2,1,figsize = (20, 10), dpi = 300)

plt.subplots_adjust(hspace = 0.5)

data.plot(ax = ax[0])
ax[0].set_title('Time series of 2015 demand and adjusted 2030 and 2040 demand for ' + country)
ax[0].set_xlabel('Time [hr]')
ax[0].set_ylabel('Demand [MW]')
ax[0].legend(loc = 'upper right')

ax[1].hist([data['2015 input demand'].values,
        data['2030 adjusted'].values, 
        data['2040 adjusted'].values],
        bins = np.arange(-4000,30000,1000), 
        rwidth=1,
        label = ['2015 input demand',
                 '2030 adjusted',
                 '2040 adjusted'])
ax[1].legend(loc = 'upper right')
ax[1].set_title(country +' Histogram of 2015 demand and adjusted 2030 and 2040 demand for ' + country)
ax[1].set_xlabel('Demand [MW]')
ax[1].set_ylabel('Frequency')

# fig.savefig('energy_adjusted.pdf', format = 'pdf', bbox_inches='tight')

