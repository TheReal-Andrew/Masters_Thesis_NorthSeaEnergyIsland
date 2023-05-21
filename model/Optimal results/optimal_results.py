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

n_2030_nac.lopf(pyomo = False,
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

n_2040_nac.lopf(pyomo = False,
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

rent_30     = get_dual(n_2030, 'Island', 'Area_Use') # [EUR/m^2]
rent_40     = get_dual(n_2040, 'Island', 'Area_Use') # [EUR/m^2]

island_price_30     = get_dual(n_2030, 'Bus', 'marginal_price')['Energy Island'] # [EUR/MW]
island_price_30_nac = get_dual(n_2030_nac, 'Bus', 'marginal_price')['Energy Island'] # [EUR/MW]
island_price_40     = get_dual(n_2040, 'Bus', 'marginal_price')['Energy Island'] # [EUR/MW]
island_price_40_nac = get_dual(n_2040_nac, 'Bus', 'marginal_price')['Energy Island'] # [EUR/MW]

prices = pd.DataFrame({'2030 price': island_price_30.copy().reset_index(drop = True),
                       '2030 nac price': island_price_30_nac.copy().reset_index(drop = True),
                       '2040 price': island_price_40.copy().reset_index(drop = True),
                       '2040_nac price': island_price_40_nac.copy().reset_index(drop = True)},
                      )
# prices = remove_outliers2(prices, prices.columns, 0)

rent_df = pd.DataFrame({'2030':rent_30,
                        '2040':rent_40},
                       )

#%% plot prices
from tabulate import tabulate

filename = 'island_electricity_prices.pdf'

fig, axs = plt.subplots(2, 1, figsize = (15,5))

prices['2030 price'].plot(ax = axs[0])

prices['2040 price'].plot(ax = axs[1], color = 'tab:blue')

fig.legend(loc = 'center',
           bbox_to_anchor = (0.5, -0.05),
          ncols = 2,
          fancybox=False, shadow=False,)

for ax, year in zip(axs, [2030, 2040]):
    ax.set(ylabel = 'Price [EUR/MW]',
            xlabel = 'Time [hr]',
            # title = f'{year}',
            )

fig.suptitle('Electricity price on the energy island')

# fig.savefig('graphics/'+filename, format = 'pdf', bbox_inches='tight')

#%% Adjusted objective values with and without area constraint

# Get value of moneybin capital cost for 2030 and 2040
moneybin_value30 = n_2030_nac.generators.capital_cost['MoneyBin']
moneybin_value40 = n_2040_nac.generators.capital_cost['MoneyBin']

# Objective function value without moneybin 
obj30     = n_2030.objective - moneybin_value30
obj30_nac = n_2030_nac.objective - moneybin_value30

obj40     = n_2040.objective - moneybin_value40
obj40_nac = n_2040_nac.objective - moneybin_value40

obj_df = pd.DataFrame( {'2030':[obj30, obj30_nac],
                        '2040':[obj40, obj40_nac]},
                      index = ['With constraint',
                               'Without constraint']
                        )

#%% Adjusted objective value without MoneyBin

# Get value of moneybin capital cost for 2030 and 2040
moneybin_value30 = n_2030_nac.generators.capital_cost['MoneyBin']
moneybin_value40 = n_2040_nac.generators.capital_cost['MoneyBin']


data_revenue30 = n_2030.generators_t.p['Data'].sum() * n_2030.generators.p_nom_opt['Data']
P2X_revenue30  = n_2030.generators_t.p['P2X'].sum() * n_2030.generators.p_nom_opt['P2X']
revenue30      = data_revenue30 + P2X_revenue30

data_revenue30_nac = n_2030_nac.generators_t.p['Data'].sum() * n_2030_nac.generators.p_nom_opt['Data']
P2X_revenue30_nac  = n_2030_nac.generators_t.p['P2X'].sum() * n_2030_nac.generators.p_nom_opt['P2X']
revenue30_nac      = data_revenue30_nac + P2X_revenue30_nac

data_revenue40 = n_2040.generators_t.p['Data'].sum() * n_2040.generators.p_nom_opt['Data']
P2X_revenue40  = n_2040.generators_t.p['P2X'].sum() * n_2040.generators.p_nom_opt['P2X']
revenue40      = data_revenue40 + P2X_revenue40

data_revenue40_nac = n_2040_nac.generators_t.p['Data'].sum() * n_2040_nac.generators.p_nom_opt['Data']
P2X_revenue40_nac  = n_2040_nac.generators_t.p['P2X'].sum() * n_2040_nac.generators.p_nom_opt['P2X']
revenue40_nac      = data_revenue40_nac + P2X_revenue40_nac

# Objective function value without moneybin 
obj30     = n_2030.objective - moneybin_value30 + abs(revenue30)
obj30_nac = n_2030_nac.objective - moneybin_value30 + abs(revenue30_nac)

obj40     = n_2040.objective - moneybin_value40 + abs(revenue30)
obj40_nac = n_2040_nac.objective - moneybin_value40 + abs(revenue40_nac)

obj_df = pd.DataFrame( {'2030':[obj30, obj30_nac],
                        '2040':[obj40, obj40_nac]},
                      index = ['With constraint',
                               'Without constraint']
                        )

#%% Adjusted values in summation loop
values = []
n_list = [n_2030, n_2030_nac, n_2040, n_2040_nac]

# countries = tm.get_bus_df()['Bus name'][1:].values

for n in n_list:
    value = 0
    
    # Capital costs
    for tech in ['P2X', 'Data']:
        
        if tech == 'Storage':
            value += n.stores.e_nom_opt[tech]* n.stores.capital_cost[tech]
        else:
            value += n.generators.p_nom_opt[tech]* n.generators.capital_cost[tech]
        
    for link in n.main_links:
        value += n.links.p_nom_opt[link] * n.links.capital_cost[link]
    
    
    # Marginal costs
    value += n.stores_t.e['Storage'].sum() * n.stores.marginal_cost['Storage']
    
    # Adjusted 
    values.append(value)
    
values_df =  pd.DataFrame( {'2030':[values[0], values[1]],
                            '2040':[values[2], values[3]]},
                             index = ['With constraint',
                                      'Without constraint']
                          )

    
#%% Network visualization
n     = n_2040
year  = 2040
title = f'{year} optimal system'
filename = f'graphics/network_diagrams/pypsa_diagram_{year}.pdf'

It = 'Island_to_'
index2 = [
          It+'Germany',
          It+'Norway',
          It+'Denmark',
          It+'Netherlands',
          It+'Belgium',
          It+'United Kingdom',
          ]

schemfig = pdiag.draw_network(n, spacing = 1, handle_bi = True,
                    index1 = index2,
                    show_country_values = False,
                    exclude_bus = 'Energy Island',
                    # filename = filename,
                    )

# Add title to ax
schemfig.ax.set_title(title, fontsize = 24)
# Get figure of schemdrawing
fig = schemfig.ax.get_figure()

# Show altered figure
display(schemfig)

# save altered figure
fig.savefig(filename, format = 'pdf', bbox_inches='tight')

#%% Waffle diagram - area and capacity for {year}

values = []
totals = []
n      = n_2040_nac
year   = 2040
title  = f'{year} optimal system without area constraint'

filename = f'waffle_{year}_area_capacity.pdf' 

col = gm.get_color_codes()
col_a = [col['P2X'], col['Data'], col['Storage']]
col_c = [col['P2X'], col['Data'], col['Storage'], col['Links']]

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

fig.suptitle(title, 
              fontsize = 28)

fig.savefig('graphics/'+filename, format = 'pdf', bbox_inches='tight')

#%% Waffle diagram nac - area and capacity for {year}

values = []
totals = []
n      = n_2030_nac
year = 2030

filename = f'waffle_{year}_nac_area_capacity.pdf' 

col = gm.get_color_codes()
col_a = [col['P2X'], col['Data'], col['Storage']]
col_c = [col['P2X'], col['Data'], col['Storage'], col['Links']]

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
                 'title':  {'label': f'Area use distribution - Area available: {int(round(total_a, -3))} $m^2$', 'loc': 'left'},
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

fig.savefig('graphics/waffles/'+filename, format = 'pdf', bbox_inches='tight')



#%% Transmission link visualization - 2030
n    = n_2040_nac
year = 2040
nac  = True
filename = f'{year}_nac_link_histograms.pdf'

flow = gm.get_link_flow(n)

if nac == True:
    flow = flow[['Denmark', 'Norway']]
    v = 3.33
elif year == 2040:
    flow = flow.drop(['Belgium', 'United Kingdom'], axis = 1)
    v = 6.66
elif year == 2030:
    flow = flow.drop('Germany', axis = 1)
    v = 10

axs = flow.hist(figsize = (20, v), bins = 100)

axs = axs.ravel()

fig = axs[0].get_figure()
fig.subplots_adjust(hspace = 0.7)
fig.suptitle(f'{year} - Link histograms for unconstrained case', fontsize = 30, y = 1.1)

for ax in axs:
    ax.set_xlabel('Power flow [MW]')
    ax.set_ylabel('Frequency')
    
fig.savefig('graphics/link_histograms/'+filename, format = 'pdf', bbox_inches='tight')

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



















