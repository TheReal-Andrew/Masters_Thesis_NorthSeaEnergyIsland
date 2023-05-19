# -*- coding: utf-8 -*-
"""
Created on Mon May 15 15:28:16 2023

@author: lukas
"""

import os
import sys
# Add modules folder to path
os.chdir(os.path.join(os.path.dirname(__file__))) # Change working directory
sys.path.append(os.path.abspath('../../modules')) # Add modules to path

import pypsa
import pandas as pd
from ttictoc import tic, toc
import matplotlib.pyplot as plt
from pywaffle import Waffle

import gorm as gm
import tim as tm
import pypsa_diagrams as pdiag

gm.set_plot_options()

#%% Prepare networks - load and insert data

year = 2030
r    = 0.07

n_2030     = pypsa.Network(r'../v_2030_local/v_2030_local_opt.nc')
n_2030_nac = pypsa.Network(r'../v_2030_local_nac/v_2030_local_nac_opt.nc')
n_2040     = pypsa.Network(r'../v_2040_local/v_2040_local_opt.nc')
n_2040_nac = pypsa.Network(r'../v_2040_local_nac/v_2040_local_nac_opt.nc')

# insert data into networks
for n in [n_2030, n_2030_nac, n_2040, n_2040_nac]:
    n.area_use      = tm.get_area_use()
    n.link_sum_max  = n.generators.p_nom_max['Wind']
    n.main_links    = n.links.loc[n.links.bus0 == "Energy Island"].index
    
#%% Run sim and get dual values

def extra_functionality(n,snapshots):
    gm.area_constraint(n, snapshots)
    gm.link_constraint(n, snapshots)
    gm.marry_links(n, snapshots)
    
n_2030.lopf(pyomo = False,
       solver_name = 'gurobi',
       keep_shadowprices = True,
       keep_references = True,
       extra_functionality = extra_functionality,
       )

n_2040.lopf(pyomo = False,
       solver_name = 'gurobi',
       keep_shadowprices = True,
       keep_references = True,
       extra_functionality = extra_functionality,
       )

#%% Get duals
from pypsa.linopt import get_dual

def remove_outliers2(df,columns,n_std):
    for col in columns:
        print('Removing outliers - Working on column: {}'.format(col))
        
        df2 = df.copy()
        
        df[col][ df[col] >= df[col].mean() + (n_std * df[col].std())] = \
        0
        
        df.mean()
        
        df2[col][ df2[col] >= df2[col].mean() + (n_std * df2[col].std())] = \
        df.mean()
        
    return df

rent_30 = get_dual(n_2030, 'Island', 'Area_Use') # [EUR/m^2]
rent_40 = get_dual(n_2040, 'Island', 'Area_Use') # [EUR/m^2]

island_price_30 = get_dual(n_2030, 'Bus', 'marginal_price')['Energy Island'] # [EUR/MW]
island_price_40 = get_dual(n_2040, 'Bus', 'marginal_price')['Energy Island'] # [EUR/MW]

prices = pd.DataFrame({'2030 price': island_price_30.copy().reset_index(drop = True),
                       '2040 price': island_price_40.copy().reset_index(drop = True)},
                      )
prices = remove_outliers2(prices, prices.columns, 0)

rent_df = pd.DataFrame({'2030':rent_30,
                        '2040':rent_40},
                       )

#%% plot prices

filename = 'island_electricity_prices.pdf'

ax = prices.plot(figsize = (10,5))

ax.legend(loc = 'center',
          bbox_to_anchor = (1.15, 0.5), fancybox=False, shadow=False,)

ax.set(ylabel = 'Electricity price [EUR/MW]',
       xlabel = 'Time [hr]',
       title  = 'Electricity price on the energy island'
       )

fig = ax.figure

fig.savefig('graphics/'+filename, format = 'pdf', bbox_inches='tight')

#%%

n = n_2030

m2_price = n.generators.capital_cost['Data']/n.area_use['data']
    
#%% Network visualization
n = n_2030

year = 2030

It = 'Island_to_'
index2 = [
          It+'Germany',
          It+'Norway',
          It+'Denmark',
          It+'Netherlands',
          It+'Belgium',
          It+'United Kingdom',
          ]


pdiag.draw_network(n, spacing = 1, handle_bi = True,
                    index1 = index2,
                    show_country_values = False,
                    exclude_bus = 'Energy Island',
                    filename = f'graphics/pypsa_diagram_{year}.pdf'
                    )

#%% 2030 Waffle diagram - area and capacity for {year}

values = []
totals = []
n      = n_2040
year = 2040

filename = f'waffle_{year}_area_capacity.pdf' 

col = gm.get_color_codes()
col_a = [col['P2X'], col['IT'], col['Storage']]
col_c = [col['P2X'], col['IT'], col['Storage'], col['Links']]

# Area data
P2X_a   = n.generators.p_nom_opt["P2X"] * n.area_use['hydrogen']
Data_a  = n.generators.p_nom_opt["Data"] * n.area_use['data']
Store_a = n.stores.e_nom_opt['Storage'] * n.area_use['storage']

total_a = P2X_a + Data_a + Store_a

val_a  = {'P2X':    (P2X_a/total_a * 100), 
          'IT':     (Data_a/total_a * 100), 
          'Storage':(Store_a/total_a * 100)}

# Capacity data
P2X_c   = n.generators.p_nom_opt["P2X"] 
Data_c  = n.generators.p_nom_opt["Data"] 
Store_c = n.stores.e_nom_opt['Storage'] 
Links_c = n.links.p_nom_opt[n.main_links].sum()


total_c = P2X_c + Data_c + Store_c + Links_c

val_c  = {'P2X':    (P2X_c/total_c * 100), 
          'IT':     (Data_c/total_c * 100), 
          'Storage':(Store_c/total_c * 100),
          'Links':  (Links_c/total_c * 100)}

# Create waffle diagram
fig = plt.figure(
    FigureClass = Waffle,
    plots = {
            311: {
                 'values': val_a,
                 'title':  {'label': f'Area use distribution - Area available: {round(total_a)} $m^2$', 'loc': 'left'},
                 'labels': [f"{k} ({int(v)}%) \n {round(v/100*total_a)} $m^2$" for k, v in val_a.items()],
                 'legend': {'loc': 'lower left', 'bbox_to_anchor': (0, -0.325), 'ncol': len(val_a), 'framealpha': 0},
                 'colors': col_a,
                  },
            312: {
                 'values': val_c,
                 'title':  {'label': f'Installed capacity distribution - Total capacity: {round(total_c/1000, 1)} $GW$', 'loc': 'left'},
                 'labels': [f"{k} ({int(v)}%) \n {round(v/100*total_c/1000, 1)} $GW$" for k, v in val_c.items()],
                 'legend': {'loc': 'lower left', 'bbox_to_anchor': (0, -0.325), 'ncol': len(val_c), 'framealpha': 0},
                 'colors': col_c,
                  },
        },
    rows   = 5,
    columns = 20,
    figsize = (10,12),
)

fig.suptitle(f'{year} optimal system', 
              fontsize = 28)

fig.savefig('graphics/'+filename, format = 'pdf', bbox_inches='tight')

#%% Transmission link visualization - 2030
n = n_2040
year = 2040
filename = f'{year}_link_histograms.pdf'

flow = gm.get_link_flow(n)

if year == 2040:
    flow = flow.drop(['Belgium', 'United Kingdom'], axis = 1)
    v = 6.66
elif year == 2030:
    flow = flow.drop('Germany', axis = 1)
    v = 10

axs = flow.hist(figsize = (20, v), bins = 100)

axs = axs.ravel()

fig = axs[0].get_figure()
fig.subplots_adjust(hspace = 0.7)
fig.suptitle(f'{year} - Link histograms', fontsize = 30)

for ax in axs:
    ax.set_xlabel('Power flow [MW]')
    ax.set_ylabel('Frequency')
    
fig.savefig('graphics/'+filename, format = 'pdf', bbox_inches='tight')

#%% Waffle diagram - area use, optimals - OLD
# filename = 'waffle_optimals_area_use.pdf' 

# values = []
# totals = []
# n_list = [n_2030, n_2040]

# col = gm.get_color_codes()

# for n in n_list:
#     P2X_a   = n.generators.p_nom_opt["P2X"] * n.area_use['hydrogen']
#     Data_a  = n.generators.p_nom_opt["Data"] * n.area_use['data']
#     Store_a = n.stores.e_nom_opt['Storage'] * n.area_use['storage']
    
    
#     total_A = P2X_a + Data_a + Store_a
    
#     val  = {'P2X':    round(P2X_a/total_A * 100), 
#             'IT':     round(Data_a/total_A * 100), 
#             'Storage':round(Store_a/total_A * 100)}
    
    
#     values.append(val)
#     totals.append(total_A)

# colors = [col['P2X'], col['IT'], col['Storage']]

# fig = plt.figure(
#     FigureClass = Waffle,
#     plots = {
#             311: {
#                  'values': values[0],
#                  'title':  {'label': f'2030 optimal system - Area available: {round(totals[0])} $m^2$', 'loc': 'left'},
#                  'labels': [f"{k} ({int(v)}%) \n {round(v/100*totals[0])} $m^2$" for k, v in values[0].items()],
#                  'legend': {'loc': 'lower left', 'bbox_to_anchor': (0, -0.325), 'ncol': len(values[0]), 'framealpha': 0},
#                  'colors': colors,
#                   },
#             312: {
#                  'values': values[1],
#                  'title':  {'label': f'2040 optimal system - Area available: {round(totals[1])} $m^2$', 'loc': 'left'},
#                  'labels': [f"{k} ({int(v)}%) \n {round(v/100*totals[1])} $m^2$" for k, v in values[1].items()],
#                  'legend': {'loc': 'lower left', 'bbox_to_anchor': (0, -0.325), 'ncol': len(values[1]), 'framealpha': 0},
#                  'colors': colors,
#                   },
#         },
#     rows   = 5,
#     columns = 20,
#     figsize = (10,12),
# )

# fig.suptitle('Area use distribution', 
#               fontsize = 28)

# fig.savefig('graphics/'+filename, format = 'pdf', bbox_inches='tight')

# #%% Waffle diagram - capacities, optimals - DONE
# filename = 'waffle_optimals_capacities.pdf'  

# values = []
# totals = []
# n_list = [n_2030, n_2040]

# col = gm.get_color_codes()

# for n in n_list:
#     P2X_a   = n.generators.p_nom_opt["P2X"]
#     Data_a  = n.generators.p_nom_opt["Data"]
#     Store_a = n.stores.e_nom_opt['Storage']
#     Links_a = n.links.p_nom_opt[n.main_links].sum()
    
#     total_A = P2X_a + Data_a + Store_a + Links_a
    
#     val  = {'P2X':    round(P2X_a/total_A * 100), 
#             'IT':     round(Data_a/total_A * 100), 
#             'Storage':round(Store_a/total_A * 100),
#             'Links':round(Links_a/total_A * 100) }
    
#     values.append(val)
#     totals.append(total_A)

# colors = [col['P2X'], col['IT'], col['Storage'], col['Links']]

# fig = plt.figure(
#     FigureClass = Waffle,
#     plots = {
#             311: {
#                  'values': values[0],
#                  'title':  {'label': f'2030 optimal system - Total capacity: {round(totals[0]/1000,1)} $GW$', 'loc': 'left'},
#                  'labels': [f"{k} ({int(v)}%) \n {round((v/100*totals[0])/1000,1)} $GW$" for k, v in values[0].items()],
#                  'legend': {'loc': 'lower left', 'bbox_to_anchor': (0, -0.325), 'ncol': len(values[0]), 'framealpha': 0},
#                  'colors': colors,
#                   },
#             312: {
#                  'values': values[1],
#                  'title':  {'label': f'2040 optimal system - Total capacity: {round(totals[1]/1000,1)} $GW$', 'loc': 'left'},
#                  'labels': [f"{k} ({int(v)}%) \n {round((v/100*totals[1])/1000,1)} $GW$" for k, v in values[1].items()],
#                  'legend': {'loc': 'lower left', 'bbox_to_anchor': (0, -0.325), 'ncol': len(values[1]), 'framealpha': 0},
#                  'colors': colors,
#                   },
#         },
#     rows   = 5,
#     columns = 20,
#     figsize = (10,12),
# )

# fig.suptitle('Capacity distribution', 
#               fontsize = 28)

# fig.savefig('graphics/'+filename, format = 'pdf', bbox_inches='tight')

# #%% Waffle diagram - area use 2040 unconstrained vs constrained

# col = gm.get_color_codes()

# filename = 'waffle_area_use_2030_unconstrained.pdf' 

# values = []
# totals = []
# n_list = [n_2030, n_2030_nac]

# year = 2030

# for n in n_list:
#     P2X_a   = n.generators.p_nom_opt["P2X"] * n.area_use['hydrogen']
#     Data_a  = n.generators.p_nom_opt["Data"] * n.area_use['data']
#     Store_a = n.stores.e_nom_opt['Storage'] * n.area_use['storage']
    
#     total_A = P2X_a + Data_a + Store_a
    
#     val  = {'P2X':    round(P2X_a/total_A * 100), 
#             'IT':     round(Data_a/total_A * 100), 
#             'Storage':round(Store_a/total_A * 100)}
    
#     values.append(val)
#     totals.append(total_A)

# values_1 = values[0]
# values_2 = values[1]

# colors = [col['P2X'], col['IT'], col['Storage']]

# fig = plt.figure(
#     FigureClass = Waffle,
#     plots = {
#             311: {
#                  'values': values_1,
#                  'title':  {'label': f'{year} optimal system', 'loc': 'left'},
#                  'labels': [f"{k} ({int(v)}%) \n {round(v/100*totals[0])} $m^2$" for k, v in values_1.items()],
#                  'legend': {'loc': 'lower left', 'bbox_to_anchor': (0, -0.3), 'ncol': len(values_1), 'framealpha': 0},
#                  'colors': colors,
#                   },
#             312: {
#                  'values': values_2,
#                  'title':  {'label': f'{year} optimal system, without area constraint', 'loc': 'left'},
#                  'labels': [f"{k} ({int(v)}%) \n {round(v/100*totals[1])} $m^2$" for k, v in values_2.items()],
#                  'legend': {'loc': 'lower left', 'bbox_to_anchor': (0, -0.3), 'ncol': len(values_2), 'framealpha': 0},
#                  'colors': colors,
#                   },
#         },
#     rows   = 5,
#     figsize = (10,11),
# )

# fig.suptitle('Area use for technologies on the island', 
#              fontsize = 20)
# # fig.set_facecolor('#fcfcfc') 
# fig.savefig('graphics/'+filename, format = 'pdf', bbox_inches='tight')

# #%% Waffle diagram - capacities 2030 unconstrained vs constrained
# col = gm.get_color_codes()

# values = []
# n_list = [n_2040, n_2040_nac]

# filename = 'waffle_capacity_2040_unconstrained.pdf'  

# year = 2040

# for n in n_list:
#     P2X_a   = n.generators.p_nom_opt["P2X"]
#     Data_a  = n.generators.p_nom_opt["Data"]
#     Store_a = n.stores.e_nom_opt['Storage']
    
#     total_A = P2X_a + Data_a + Store_a
    
#     val  = {'P2X':    round(P2X_a/total_A * 100), 
#             'IT':     round(Data_a/total_A * 100), 
#             'Storage':round(Store_a/total_A * 100)}
    
#     values.append(val)

# values_1 = values[0]
# values_2 = values[1]

# colors = [col['P2X'], col['IT'], col['Storage']]

# fig = plt.figure(
#     FigureClass = Waffle,
#     plots = {
#             311: {
#                  'values': values_1,
#                  'title':  {'label': f'{year} optimal system', 'loc': 'left'},
#                  'labels': [f"{k} ({v}%)" for k, v in values_1.items()],
#                  'legend': {'loc': 'lower left', 'bbox_to_anchor': (0, -0.3), 'ncol': len(values_1), 'framealpha': 0},
#                  'colors': colors,
#                   },
#             312: {
#                  'values': values_2,
#                  'title':  {'label': f'{year} optimal system, without area constraint', 'loc': 'left'},
#                  'labels': [f"{k} ({v}%)" for k, v in values_2.items()],
#                  'legend': {'loc': 'lower left', 'bbox_to_anchor': (0, -0.3), 'ncol': len(values_2), 'framealpha': 0},
#                  'colors': colors,
#                   },
#         },
#     rows   = 5,
#     figsize = (10,11),
# )

# fig.suptitle('Installed capacities of technologies on the island', 
#              fontsize = 20)
# # fig.set_facecolor('#fcfcfc')
# fig.savefig('graphics/'+filename, format = 'pdf', bbox_inches='tight')

#%% Waffle diagram - capacities unconstrained vs constrained

col = gm.get_color_codes()

values = []
n_list = [n_2030, n_2030_nac]

year = 2030

for n in n_list:
    P2X_a   = n.generators.p_nom_opt["P2X"]
    Data_a  = n.generators.p_nom_opt["Data"]
    Store_a = n.stores.e_nom_opt['Storage']
    
    total_A = P2X_a + Data_a + Store_a
    
    val  = {'P2X':    round(P2X_a/total_A * 100), 
            'IT':     round(Data_a/total_A * 100), 
            'Storage':round(Store_a/total_A * 100)}
    
    values.append(val)

values_1 = values[0]
values_2 = values[1]

colors = [col['P2X'], col['IT'], col['Storage']]

fig = plt.figure(
    FigureClass = Waffle,
    plots = {
            311: {
                 'values': values_1,
                 'title':  {'label': f'{year} optimal system', 'loc': 'left'},
                 'labels': [f"{k} ({v}%)" for k, v in values_1.items()],
                 'legend': {'loc': 'lower left', 'bbox_to_anchor': (0, -0.3), 'ncol': len(values_1), 'framealpha': 0},
                 'colors': colors,
                  },
            312: {
                 'values': values_2,
                 'title':  {'label': f'{year} optimal system, without area constraint', 'loc': 'left'},
                 'labels': [f"{k} ({v}%)" for k, v in values_2.items()],
                 'legend': {'loc': 'lower left', 'bbox_to_anchor': (0, -0.3), 'ncol': len(values_2), 'framealpha': 0},
                 'colors': colors,
                  },
        },
    rows   = 5,
    figsize = (10,11),
)

fig.suptitle('Installed capacities of technologies on the island', 
             fontsize = 20)
fig.set_facecolor('#fcfcfc')


















#%% 



















