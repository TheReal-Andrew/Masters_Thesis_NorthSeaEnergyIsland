# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:38:14 2023

@author: lukas
"""

import pandas as pd
import gorm as gm
import matplotlib.pyplot as plt

gm.set_plot_options()

country = 'BE'

data30   = pd.read_csv(f'../data/market/demand_2030.csv', index_col = 0)

data40   = pd.read_csv(f'../data/market/demand_2040.csv', index_col = 0)

data_a30 = pd.read_csv(f'../data/market/el_demand_adjusted_2030.csv', index_col = 0)

data_a40 = pd.read_csv(f'../data/market/el_demand_adjusted_2040.csv', index_col = 0)

data_a50 = pd.read_csv(f'../data/market/el_demand_adjusted_2050.csv', index_col = 0)

data = pd.concat([data30[country], data40[country], data_a30[country], data_a40[country],
                  # data_a50['DK']],
                 ], axis = 1)

data.columns = ['2030 input', '2040 input', '2030 adjusted', '2040 adjusted',
                # 'DK_2050_a',
                ]
data.index = pd.to_datetime(data.index)

ax = data.plot(figsize = (15,7))

fig = plt.gcf()

plt.suptitle( country +' projected energy consumption for 2030 and 2040')
ax.set_title('PES30P input demand vs demand adjusted for electrification', fontsize = 16)
ax.set_xlabel('Time [hr]')
ax.set_ylabel('Demand [MW]')


# fig.savefig('energy adjusted.pdf', format = 'pdf', bbox_inches='tight')

