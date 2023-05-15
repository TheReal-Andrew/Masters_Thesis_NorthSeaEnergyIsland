# -*- coding: utf-8 -*-
"""
Created on Fri May 12 15:12:45 2023

@author: lukas
"""

import os
import sys
# Add modules folder to path
os.chdir(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.abspath('../../modules')) 

import pandas as pd
import gorm as gm
import tim as tm
import matplotlib.pyplot as plt
import numpy as np

country = 'Denmark'

cprice, cload30   = tm.get_load_and_price(2030, n_std = 1)
cload30 = cload30.applymap(lambda x: 0 if x < 0 else x) # Remove negative values

cprice, cload40   = tm.get_load_and_price(2040, n_std = 1)
cload40 = cload40.applymap(lambda x: 0 if x < 0 else x) # Remove negative values

cload15 = pd.read_csv(f'../../data/market/demand_2030.csv', index_col = 0)

a = cload15.loc[:, 'DK'].rename('DK_2015')
b = cload30.loc[:, 'DK'].rename('DK_2030')
c = cload40.loc[:, 'DK'].rename('DK_2040')

data = pd.concat([a, b, c], axis=1)

fig, ax = plt.subplots(2,1,figsize = (20, 10), dpi = 300)

plt.subplots_adjust(hspace = 0.5)

data.plot(ax = ax[0])
ax[0].set_title('Time series of 2015 demand and adjusted 2030 and 2040 demand for ' + country)
ax[0].set_xlabel('Time [hr]')
ax[0].set_ylabel('Demand [MW]')
ax[0].legend(loc = 'upper right')

ax[1].hist([data['DK_2015'].values,
        data['DK_2030'].values, 
        data['DK_2040'].values],
        bins = np.arange(-4000,30000,1000), 
        rwidth=1,
        label = ['2015 input demand',
                 '2030 adjusted',
                 '2040 adjusted'])
ax[1].legend(loc = 'upper right')
ax[1].set_title('Histogram of 2015 demand and adjusted 2030 and 2040 demand for ' + country)
ax[1].set_xlabel('Demand [MW]')
ax[1].set_ylabel('Frequency')

fig.savefig('energy_adjusted.pdf', format = 'pdf', bbox_inches='tight')