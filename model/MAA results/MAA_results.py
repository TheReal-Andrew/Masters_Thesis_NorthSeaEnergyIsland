# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:00:12 2023

@author: lukas
"""

import os
import sys
# Add modules folder to path
os.chdir(os.path.join(os.path.dirname(__file__))) # Change working directory
sys.path.append(os.path.abspath('../../modules')) # Add modules to path

import pypsa
import pandas as pd
import numpy as np
from ttictoc import tic, toc
import matplotlib.pyplot as plt
from pywaffle import Waffle

import gorm as gm
import tim as tm
import pypsa_diagrams as pdiag

gm.set_plot_options()

#%% Load all solutions

# Local solutions
project = 'local'
n_MAA   = 3
local30     = np.load(f'../v_2030_{project}/v_2030_{project}_{n_MAA}MAA_10p_solutions.npy')
local30_nac = np.load(f'../v_2030_{project}_nac/v_2030_{project}_nac_{n_MAA}MAA_10p_solutions.npy')
local40     = np.load(f'../v_2040_{project}/v_2040_{project}_{n_MAA}MAA_10p_solutions.npy')
local40_nac = np.load(f'../v_2040_{project}_nac/v_2040_{project}_nac_{n_MAA}MAA_10p_solutions.npy')

# Link solutions
linksG130 = np.load('../v_2030_links_G1/v_2030_links_G1_4MAA_10p_solutions.npy')
linksG230 = np.load('../v_2030_links_G2/v_2030_links_G2_3MAA_10p_solutions.npy')
linksG330 = np.load('../v_2030_links_G3/v_2030_links_G3_3MAA_10p_solutions.npy')
linksG140 = np.load('../v_2040_links_G1/v_2040_links_G1_4MAA_10p_solutions_temp.npy')
linksG240 = np.load('../v_2040_links_G2/v_2040_links_G2_3MAA_10p_solutions.npy')
linksG340 = np.load('../v_2040_links_G3/v_2040_links_G3_3MAA_10p_solutions.npy')

# optimal systems
local30_opt     = pypsa.Network('../v_2030_local/v_2030_local_opt.nc')
local30_opt_nac = pypsa.Network('../v_2030_local_nac/v_2030_local_nac_opt.nc')
local40_opt     = pypsa.Network('../v_2040_local/v_2040_local_opt.nc')
local40_opt_nac = pypsa.Network('../v_2040_local_nac/v_2040_local_nac_opt.nc')

linksG130_opt   = pypsa.Network('../v_2030_links_G1/v_2030_links_G1_opt.nc')
linksG230_opt   = pypsa.Network('../v_2030_links_G2/v_2030_links_G2_opt.nc')
linksG330_opt   = pypsa.Network('../v_2030_links_G3/v_2030_links_G3_opt.nc')

linksG140_opt   = pypsa.Network('../v_2040_links_G1/v_2040_links_G1_opt.nc')
linksG240_opt   = pypsa.Network('../v_2040_links_G2/v_2040_links_G2_opt.nc')
linksG340_opt   = pypsa.Network('../v_2040_links_G3/v_2040_links_G3_opt.nc')

techs_list  = [['P2X', 'Data', 'Storage'], 
               ['P2X', 'Data', 'Storage'],
               ['P2X', 'Data', 'Storage'],
               ['P2X', 'Data', 'Storage'],
               ['DK', 'DE', 'NL', 'BE'],   
               ['DE', 'NL', 'GB'],         
               ['DK', 'NO', 'BE'],        
               ['DK', 'DE', 'NL', 'BE'],   
               ['DE', 'NL', 'GB'],         
               ['DK', 'NO', 'BE'],
               ]

n_opt_list  = [local30_opt,   
               local30_opt_nac,
               local40_opt,   
               local40_opt_nac,
               linksG130_opt, 
               linksG230_opt,
               linksG330_opt,
               linksG140_opt,
               linksG240_opt,
               linksG340_opt,
               ] 

case_list = [local30,   
             local30_nac,
             local40,   
             local40_nac,
             linksG130, 
             linksG230,
             linksG330,
             linksG140,
             linksG240,
             linksG340,
             ] 

opt_list = []

bus_df = tm.get_bus_df()

i = 0
for n_opt in n_opt_list:
    
    if i < 4:
        variables = ['P2X', 'Data', 'Storage']
        
        opt_system = n_opt.generators.loc[n_opt.generators.index.isin(variables)].p_nom_opt.tolist()
        opt_system.append(n_opt.stores.loc[n_opt.stores.index.isin(variables)].e_nom_opt['Storage'])
        
    elif i == 4 or i == 7:
        variables = ['DK', 'DE', 'NL', 'BE']
        
        
        countries  = bus_df.loc[bus_df['Abbreviation'].isin(variables)]['Bus name'].tolist()
        opt_system = n_opt.links[n_opt.links['bus0'].isin(countries)].p_nom_opt.tolist()
        
    elif i == 5 or i == 8:
        variables =  ['DE', 'NL', 'GB']
        
        countries  = bus_df.loc[bus_df['Abbreviation'].isin(variables)]['Bus name'].tolist()
        opt_system = n_opt.links[n_opt.links['bus0'].isin(countries)].p_nom_opt.tolist()
        
    elif i == 6 or i == 9:
        variables =  ['DK','NO','BE']
        
        countries  = bus_df.loc[bus_df['Abbreviation'].isin(variables)]['Bus name'].tolist()
        opt_system = n_opt.links[n_opt.links['bus0'].isin(countries)].p_nom_opt.tolist()
        
    opt_list.append(opt_system)
    
    i += 1
    
#%% Investigate links MAA slack
n             = linksG340_opt
n.main_links  = n.links.loc[n.links.bus0 == "Energy Island"].index

p_noms = n.links.loc[n.main_links]['p_nom_opt'] 
ccs    = n.links.loc[n.main_links]['capital_cost']
    
costs = p_noms * ccs

slack = 0.1 * costs.sum()

#%%

sol = np.load('../v_2040_links_G2/v_2040_links_G2_3MAA_10p_solutions.npy')
techs = ['DE', 'NL', 'GB']

gm.solutions_2D(techs, sol, n_samples = 10_000, title = '2040 Links G2')

#%% Solutions_2D for all studies

projects = ['local', 'local_nac', 
            'local', 'local_nac', 
            'links_G1', 'links_G2', 'links_G3',
            'links_G1', 'links_G2', 'links_G3',
            ]

titles   = ['MAA results 2030 - Local demand', 'MAA results, 2030 - Unconstrained local demand',
               'MAA results 2040 - Local demand', 'MAA results, 2040 - Unconstrained local demand',
               'MAA results 2030 - Planned links', 'MAA results 2030 - Low capacity links', 'MAA results 2030 - High capacity links',
               'MAA results 2040 - Planned links', 'MAA results 2040 - Low capacity links', 'MAA results 2040 - High capacity links',
               ]

years    = [2030, 2030, 
            2040, 2040,
            2030, 2030, 2030, 
            2040, 2040, 2040,]



for i in range(len(case_list)):
    
    filename = f'graphics/v_{years[i]}_{projects[i]}_{len(techs_list[i])}MAA_10p_plot_2D_MAA.pdf'
    
    gm.solutions_2D(techs_list[i], case_list[i],
                    n_samples = 10_000,
                    # filename = filename,
                    title = titles[i],
                    opt_system = opt_list[i]
                    )
    
#%% 
i = 7

filename = f'graphics/v_{years[i]}_{projects[i]}_{len(techs_list[i])}MAA_10p_plot_2D_MAA.pdf'

gm.solutions_2D(techs_list[i], case_list[i],
                n_samples = 1_000_000,
                filename = filename,
                title = titles[i],
                opt_system = opt_list[i]
                )
gm.solutions_2D(techs)


#%% Load solutions

project = 'local'
n_MAA   = 3

# solutions = np.load('../v_2030_links_G2/v_2030_links_G2_3MAA_10p_solutions.npy')
# techs     = ['DE', 'NL', 'GB']

# solutions = np.load('../v_2030_links_G3/v_2030_links_G3_3MAA_10p_solutions.npy')
# techs     = ['DK', 'NO', 'BE']

solutions30 = local30
solutions40 = local40
techs     = ['P2X', 'Data', 'Storage']

n_samples    = 100000
bins         = 50
density_vars = ['P2X', 'Data']
colors       = gm.get_color_codes()
opt_system = None

#%% Overlapping histograms

n_samples   = 100_000
tech_colors = gm.get_color_codes()

# project, n_MAA, techs, title

projects    = ['local', 'local_nac', 'links_G1', 'links_G2', 'links_G3']
techs_list  = [['P2X', 'Data', 'Storage'],
               ['P2X', 'Data', 'Storage'],
               ['DK', 'DE', 'NL', 'BE'],
               ['DE', 'NL', 'GB'],
               ['DK', 'NO', 'BE'],]

titles      = ['MAA solution histograms for local demand',
               'MAA solution histograms for unconstrained local demand',
               'MAA solution histograms for planned links',
               'MAA solution histograms for low capacity links',
               'MAA solution histograms for high capacity links',
               ]

i = 0
for techs, project in zip(techs_list, projects):
    #Local:
    filename = f'MAA_overlap_hist_{project}.pdf'
    n_MAA   = len(techs)
    sol30 = np.load(f'../v_2030_{project}/v_2030_{project}_{n_MAA}MAA_10p_solutions.npy')
    sol40 = np.load(f'../v_2040_{project}/v_2040_{project}_{n_MAA}MAA_10p_solutions.npy')
    sols  = [sol30, sol40]
    title = titles[i]
    
    if i == 2:
        y = 0.933
    else:
        y = 0.96
    
    gm.histograms_3MAA(techs, sols, title = title, n_samples = n_samples,
                       titlesize = 20, titley = y,
                        filename = filename,
                       )
    
    i += 1
    
    

#%%
#Local:
project = 'local'
n_MAA   = 3
solutions30 = np.load(f'../v_2030_{project}/v_2030_{project}_{n_MAA}MAA_10p_solutions.npy')
solutions40 = np.load(f'../v_2040_{project}/v_2040_{project}_{n_MAA}MAA_10p_solutions.npy')

solutions_list = [solutions30,
                   solutions40,
                  ]
    
gm.histograms_3MAA(techs, solutions_list, n_samples = n_samples)

#%% Histogram + MAA density function
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull

fig, axs = plt.subplots(1, 2, figsize = (20,5),
                       gridspec_kw={'width_ratios': [1, 3]},
                      )

for solutions in [solutions30, solutions40]:
    # Solutions to df
    solutions_df = pd.DataFrame(data = solutions,
                                columns = techs)
    
    # Sampling
    samples = sample_in_hull(solutions, n_samples)
    samples_df = pd.DataFrame(samples, columns = techs)
    
    #
    
    # ----------------------- Histograms plot -----------------------------
    for tech in techs:
            
        ax = axs[1]
        
        sns.histplot(samples_df[tech].values, 
                     line_kws = {'linewidth':3},
                     color = colors[tech],
                     alpha = 1/len(techs*2),
                     kde = True,
                     ax = ax, label = tech,
                     )
            
    axs[1].legend(bbox_to_anchor=(0.5, -0.25), loc = 'center', 
                  ncol = len(techs), fancybox=False, shadow=False,)
    axs[1].set(xlabel = 'Installed capacity [MW]/[MWh]', ylabel = 'Frequency')
    
    #%
    # MAA density plot
    
    x, y = solutions_df[density_vars[0]], solutions_df[density_vars[1]]
    x_samples, y_samples = samples_df[density_vars[0]], samples_df[density_vars[1]]
    
    # --------  Create 2D histogram --------------------
    hist, xedges, yedges = np.histogram2d(x_samples, y_samples,
                                          bins = bins)
    
    # Create grid for pcolormesh
    X, Y = np.meshgrid(xedges, yedges)
    
    # Create pcolormesh plot with square bins
    axs[0].pcolormesh(X, Y, hist.T, cmap = 'Blues', zorder = 0)
    
    # Create patch to serve as hexbin label
    hb = mpatches.Patch(color = 'tab:blue')
    
    # Set labels and grid
    xlabel = 'IT' if density_vars[0] == 'Data' else density_vars[0] 
    ylabel = 'IT' if density_vars[1] == 'Data' else density_vars[1]
    xUnit  = ' [MWh]' if density_vars[0] == 'Storage' else ' [MW]'
    yUnit  = ' [MWh]' if density_vars[1] == 'Storage' else ' [MW]'
    
    axs[0].grid('on')
    axs[0].set(xlabel = xlabel + xUnit, ylabel = ylabel + yUnit)
    
    # --------  Plot hull --------------------
    hull = ConvexHull(solutions_df[density_vars])
    
    solutions_plot = solutions_df[density_vars].values
    
    # plot simplexes
    for simplex in hull.simplices:
        l0, = axs[0].plot(solutions_df[density_vars[0]][simplex],
                          solutions_df[density_vars[1]][simplex],
                          '-', color = 'silver', label = 'faces', zorder = 0)
        
    l_list, l_labels   = [l0, hb], ['Polyhedron face', 'Sample density']
        
    # # optimal solutions
    # if not opt_system == None:
    #     x_opt, y_opt = opt_system[i],   opt_system[j]
        
    #     # Plot optimal solutions
    #     l2, = ax.plot(x_opt, y_opt,
    #               'o', label = "Optimal", 
    #               ms = 20, color = 'red',
    #               zorder = 3)
        
    #     l_list.append(l2)
    #     l_labels.append('Optimal solution')
        
        
    ncols = len(techs) if opt_system == None else len(techs)+1
    axs[0].legend(l_list, l_labels, 
                  bbox_to_anchor=(0.5, -0.25),
                  loc = 'center', ncol = ncols,fancybox=False, shadow=False,)


#%%



















