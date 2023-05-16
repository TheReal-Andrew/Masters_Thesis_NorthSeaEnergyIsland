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

#%% 2030 results

year = 2030
r    = 0.07

n_2030     = pypsa.Network(r'../v_2030_local/v_2030_local_opt.nc')
n_2030_nac = pypsa.Network(r'../v_2030_local_nac/v_2030_local_nac_opt.nc')

# insert data into networks
for n in [n_2030, n_2030_nac]:
    n.area_use      = tm.get_area_use()
    n.link_sum_max  = n.generators.p_nom_max['Wind']
    n.main_links    = n.links.loc[n.links.bus0 == "Energy Island"].index


#%%
for n, year in zip([n_2030, n_2030_nac], ['2030', '2030_nac']):
    fig, axs = plt.subplots(1, 2, figsize = (16,8))
        
    labels = ['P2X', 'Data', 'Storage', 'Links']
    
    gm.bake_local_area_pie(n, ax = axs[0],
                           title = f'Area use distribution - {year}',
                           )
    gm.bake_capacity_pie(n, ax = axs[1],
                         title = f'Capacity distribution - {year}',
                         )
    
    plt.legend(labels = labels, 
              loc = 'center', ncol = 4,
              bbox_to_anchor=(-0.1, -0.05), fancybox=False, shadow=False,)

#%% 2040 results

year = 2040
r    = 0.07

n_2040     = pypsa.Network(r'../v_2040_local/v_2040_local_opt.nc')
n_2040_nac = pypsa.Network(r'../v_2040_local_nac/v_2040_local_nac_opt.nc')

# insert data into networks
for n in [n_2040, n_2040_nac]:
    n.area_use      = tm.get_area_use()
    n.link_sum_max  = n.generators.p_nom_max['Wind']
    n.main_links    = n.links.loc[n.links.bus0 == "Energy Island"].index

#%%
for n, year in zip([n_2030, n_2030_nac], ['2040', '2040_nac']):
    fig, axs = plt.subplots(1, 2, figsize = (16,8))
        
    labels = ['P2X', 'Data', 'Storage', 'Links']
    
    gm.bake_local_area_pie(n, ax = axs[0],
                           title = f'Area use distribution - {year}',
                           )
    gm.bake_capacity_pie(n, ax = axs[1],
                         title = f'Capacity distribution - {year}',
                         )
    
    plt.legend(labels = labels, 
              loc = 'center', ncol = 4,
              bbox_to_anchor=(-0.1, -0.05), fancybox=False, shadow=False,)
    
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

    
#%% Waffle diagram - area use
values   = []
filename = 'waffle_optimals_area_use.pdf' 

col = gm.get_color_codes()

for n in [n_2030, n_2040]:
    P2X_a   = n.generators.p_nom_opt["P2X"] * n.area_use['hydrogen']
    Data_a  = n.generators.p_nom_opt["Data"] * n.area_use['data']
    Store_a = n.stores.e_nom_opt['Storage'] * n.area_use['storage']
    
    total_A = P2X_a + Data_a + Store_a
    
    val  = {'P2X':    round(P2X_a/total_A * 200), 
            'IT':     round(Data_a/total_A * 200), 
            'Storage':round(Store_a/total_A * 200)}
    
    values.append(val)

values_30 = values[0]
values_40 = values[1]

colors = [col['P2X'], col['IT'], col['Storage']]

fig = plt.figure(
    FigureClass = Waffle,
    plots = {
            311: {
                 'values': values_30,
                 'title':  {'label': '2030 optimal system', 'loc': 'left'},
                 'labels': [f"{k} ({v/2}%)" for k, v in values_30.items()],
                 'legend': {'loc': 'lower left', 'bbox_to_anchor': (0, -0.3), 'ncol': len(values_30), 'framealpha': 0},
                 'colors': colors,
                  },
            312: {
                 'values': values_40,
                 'title':  {'label': '2040 optimal system', 'loc': 'left'},
                 'labels': [f"{k} ({v/2}%)" for k, v in values_40.items()],
                 'legend': {'loc': 'lower left', 'bbox_to_anchor': (0, -0.3), 'ncol': len(values_40), 'framealpha': 0},
                 'colors': colors,
                  },
        },
    rows   = 10,
    figsize = (10,11),
)

fig.suptitle('Area use distribution', 
             fontsize = 28)
# fig.set_facecolor('#fcfcfc')

fig.savefig('graphics/'+filename, format = 'pdf', bbox_inches='tight')

#%% Waffle diagram - capacities
col = gm.get_color_codes()

filename = 'waffle_optimals_capacities.pdf' 

values = []
n_list = [n_2030, n_2040]

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
                 'title':  {'label': '2030 optimal system', 'loc': 'left'},
                 'labels': [f"{k} ({v}%)" for k, v in values_1.items()],
                 'legend': {'loc': 'lower left', 'bbox_to_anchor': (0, -0.3), 'ncol': len(values_1), 'framealpha': 0},
                 'colors': colors,
                  },
            312: {
                 'values': values_2,
                 'title':  {'label': '2040 optimal system', 'loc': 'left'},
                 'labels': [f"{k} ({v}%)" for k, v in values_2.items()],
                 'legend': {'loc': 'lower left', 'bbox_to_anchor': (0, -0.3), 'ncol': len(values_2), 'framealpha': 0},
                 'colors': colors,
                  },
        },
    rows   = 5,
    figsize = (10,11),
)

fig.suptitle('Capacity distribution', 
             fontsize = 28)
# fig.set_facecolor('#fcfcfc')
fig.savefig('graphics/'+filename, format = 'pdf', bbox_inches='tight')

#%% Waffle diagram - area use 2040 unconstrained vs constrained

col = gm.get_color_codes()

filename = 'waffle_area_use_2030_unconstrained.pdf' 

values = []
n_list = [n_2030, n_2030_nac]

year = 2030

for n in n_list:
    P2X_a   = n.generators.p_nom_opt["P2X"] * n.area_use['hydrogen']
    Data_a  = n.generators.p_nom_opt["Data"] * n.area_use['data']
    Store_a = n.stores.e_nom_opt['Storage'] * n.area_use['storage']
    
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

fig.suptitle('Area use for technologies on the island', 
             fontsize = 20)
# fig.set_facecolor('#fcfcfc') 
fig.savefig('graphics/'+filename, format = 'pdf', bbox_inches='tight')

#%% Waffle diagram - capacities 2030 unconstrained vs constrained
col = gm.get_color_codes()

values = []
n_list = [n_2040, n_2040_nac]

filename = 'waffle_capacity_2040_unconstrained.pdf'  

year = 2040

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
# fig.set_facecolor('#fcfcfc')
fig.savefig('graphics/'+filename, format = 'pdf', bbox_inches='tight')

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




















