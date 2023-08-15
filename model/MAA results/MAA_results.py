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
import polytope

import gorm as gm
import tim as tm
import pypsa_diagrams as pdiag

gm.set_plot_options()

#%% Load eeeeeverything. Solutions and optimal systems.

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

projects = ['local', 'local_nac', #2030
            'local', 'local_nac', #2040
            'links_G1', 'links_G2', 'links_G3', #2030
            'links_G1', 'links_G2', 'links_G3', #2040
            ]

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
             linksG130,  #
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
    
#%% Link time series

n = local30_opt
year = 2030
filename = f'{year}_stacked_area_plot.pdf'

# Define links that go to and from island
main_links1          = n.links.loc[n.links.bus0 == "Energy Island"].index
main_links2          = n.links.loc[n.links.bus1 == "Energy Island"].index

# Get the list of column names
column_order = main_links1.tolist()

# Rearrange the column order as desired
main_links1   = [column_order[5], 
                 column_order[4], 
                 column_order[3], 
                 column_order[2],
                 column_order[0],
                 column_order[1]]

# Get the list of column names
column_order = main_links2.tolist()

# Rearrange the column order as desired
main_links2   = [column_order[5], 
                 column_order[4], 
                 column_order[3], 
                 column_order[2],
                 column_order[0],
                 column_order[1]]

colors = gm.get_color_codes()

country_abbreviations = []
color_codes = []

for key, value in colors.items():
    if len(key) == 2:  # Check if key is a country abbreviation
        country_abbreviations.append(key)
        color_codes.append(value)
        
color_codes.reverse()
color_codes[4], color_codes[5] = color_codes[5], color_codes[4]


fig, axs = plt.subplots(2, 1, figsize = (15,5))
fig.subplots_adjust(hspace = 0.45)

sending_df   = n.links_t.p0[main_links1].resample('D').mean()/1000
recieving_df = n.links_t.p1[main_links2].resample('D').mean()/1000

sending_df.columns = ['GB', 'BE', 'NL', 'DE', 'DK', 'NO']
recieving_df.columns = ['GB', 'BE', 'NL', 'DE', 'DK', 'NO']

# sending_df.index = sending_df.index - pd.DateOffset(years=1)
# recieving_df.index = recieving_df.index - pd.DateOffset(years=1)

# Time series of island to country link, positive when island is sending
sending_df.plot.area(figsize = (20,10), ax = axs[0],
                              title = f'{year} - Flow from island to countries in',
                              color = color_codes)

# Time series of country to island link, negative when island is recieving
recieving_df.plot.area(figsize = (20,10), ax = axs[1],
                               title = f'{year} - Flow from countries to island',
                               color = color_codes,
                               )

# Plot formatting
legend0 = axs[0].legend(loc='center left', 
                        bbox_to_anchor=(1, 0.5))
legend1 = axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

axs[0].set_xlabel('Time [Day]')
axs[1].set_xlabel('Time [Day]')
axs[0].set_ylabel('Flow [GW]')
axs[1].set_ylabel('Flow [GW]')


fig.savefig('graphics/'+filename, format = 'pdf', bbox_inches='tight')

#%% Create tables with relevant data

for i in range(len(case_list)):

    if i in [0, 1] or i in [4, 5, 6]:
        year = 2030
    else:
        year = 2040
    
    # Get data for this project
    project   = projects[i]
    
    techs     = techs_list[i]
    
    solutions = case_list[i]
    opt       = opt_list[i]
    
    
    # Optimal
    opt_df = pd.DataFrame(np.array([opt]), columns = techs,
                          index = ['Optimum'])
    
    # Chevyshev center
    p1 = polytope.qhull(solutions)
    cheb_center = p1.chebXc # Chebyshev ball center
    cheb_radius = p1.chebR
    
    cheb_df = pd.DataFrame(np.array([cheb_center]), columns = techs,
                           index = ['Chebyshev'])
    
    # Minmax combinations
    solutions_df = pd.DataFrame(solutions, columns = techs)
    dfs = []
    
    for tech in techs:
        large = solutions_df.nlargest(1, tech)
        large.index = ['max ' + tech]
        
        small = solutions_df.nsmallest(1, tech)
        small.index = ['min ' + tech]
        
        dfs.append(large)
        dfs.append(small)
               

    # Create dataframe from all results                        
    minmax_df = pd.concat(dfs)
        
    results_df = pd.concat([opt_df, cheb_df, minmax_df])
    
    # Export to Latex
    latex = results_df.to_latex(float_format = '%.0f')
    
    filename = os.path.join('tables', f"{year}_{project}_table.txt")
    
    # Write the LaTeX code to the text file
    with open(filename, 'w') as file:
        file.write(latex)


        

#%% Solutions_2D_small

n_samples = 1000_000
bins = 150
legend_v = -0.47
legend_h = 0.9


# Solutions 2D Small for DE/NL in G2 2030 -------------------------------------
studyno = 5 
solutions = case_list[studyno]
techs     = techs_list[studyno]
opt       = opt_list[studyno]

# Convert to GW
opt = [x/1000 for x in opt]
solutions = solutions/1000

chosen_techs = ['DE', 'NL']
year  = 2030
study = 'G2'
title = f'{year} - MAA and histogram plot for MAA variables: {chosen_techs[0]}, {chosen_techs[1]}'

axs = gm.solutions_2D_small(techs, solutions,
                       # axs = axs,
                      chosen_techs = chosen_techs,
                      cheb = True, opt_system = opt,
                      show_minmax = False,
                      n_samples = n_samples,
                      legend_v = legend_v, legend_h = legend_h,
                      figsize = (20,4),
                      title = title,
                      hist_bins = bins,
                      )

fig = axs[0].get_figure()
filename = f'v_{year}_{study}_{chosen_techs[0]}-{chosen_techs[1]}_MAA_small.pdf'
fig.savefig(f'graphics/{filename}', format = 'pdf', bbox_inches='tight')




# Solutions 2D Small for DE/NL in G2 2040 -------------------------------------
studyno = 5 + 3
solutions = case_list[studyno]
techs     = techs_list[studyno]
opt       = opt_list[studyno]

# Convert to GW
opt = [x/1000 for x in opt]
solutions = solutions/1000

chosen_techs = ['DE', 'NL']
year  = 2040
study = 'G2'
title = f'{year} - MAA and histogram plot for MAA variables: {chosen_techs[0]}, {chosen_techs[1]}'

axs = gm.solutions_2D_small(techs, solutions,
                       # axs = axs,
                      chosen_techs = chosen_techs,
                      cheb = True, opt_system = opt,
                      show_minmax = False,
                      n_samples = n_samples,
                      legend_v = legend_v, legend_h = legend_h,
                      figsize = (20,4),
                      title = title,
                      hist_bins = bins,
                      )

fig = axs[0].get_figure()
filename = f'v_{year}_{study}_{chosen_techs[0]}-{chosen_techs[1]}_MAA_small.pdf'
fig.savefig(f'graphics/{filename}', format = 'pdf', bbox_inches='tight')




# Solutions 2D small for DK/BE in G3 2030 ------------------------------------
studyno = 6
solutions = case_list[studyno]
techs     = techs_list[studyno]
opt       = opt_list[studyno]

# Convert to GW
opt = [x/1000 for x in opt]
solutions = solutions/1000

chosen_techs = ['DK', 'BE']
year  = 2030
study = 'G3'
title = f'{year} - MAA and histogram plot for MAA variables: {chosen_techs[0]}, {chosen_techs[1]}'

axs = gm.solutions_2D_small(techs, solutions,
                       # axs = axs,
                      chosen_techs = chosen_techs,
                      cheb = True, opt_system = opt,
                      show_minmax = False,
                      n_samples = n_samples,
                      legend_v = legend_v, legend_h = legend_h,
                      figsize = (20,4),
                      title = title,
                      hist_bins = bins,
                      )

fig = axs[0].get_figure()
filename = f'v_{year}_{study}_{chosen_techs[0]}-{chosen_techs[1]}_MAA_small.pdf'
fig.savefig(f'graphics/{filename}', format = 'pdf', bbox_inches='tight')




# Solutions 2D small for DK/BE in G3 2040 ------------------------------------
studyno = 6 + 3
solutions = case_list[studyno]
techs     = techs_list[studyno]
opt       = opt_list[studyno]

# Convert to GW
opt = [x/1000 for x in opt]
solutions = solutions/1000

chosen_techs = ['DK', 'BE']
year  = 2040
study = 'G3'
title = f'{year} - MAA and histogram plot for MAA variables: {chosen_techs[0]}, {chosen_techs[1]}'

axs = gm.solutions_2D_small(techs, solutions,
                       # axs = axs,
                      chosen_techs = chosen_techs,
                      cheb = True, opt_system = opt,
                      show_minmax = False,
                      n_samples = n_samples,
                      legend_v = legend_v, legend_h = legend_h,
                      figsize = (20,4),
                      title = title,
                      hist_bins = bins,
                      )

fig = axs[0].get_figure()
filename = f'v_{year}_{study}_{chosen_techs[0]}-{chosen_techs[1]}_MAA_small.pdf'
fig.savefig(f'graphics/{filename}', format = 'pdf', bbox_inches='tight')




# Solutions 2D small for Data/Storage in Local 2030 -----------------------------
studyno = 0
solutions = case_list[studyno]
techs     = techs_list[studyno]
opt       = opt_list[studyno]

# Convert to GW
opt = [x/1000 for x in opt]
solutions = solutions/1000

chosen_techs = ['Data', 'Storage']
year  = 2030
study = 'local'
tech_names = ['IT' if tech == 'Data' else tech for tech in chosen_techs]
title = f'{year} - MAA and histogram plot for MAA variables: {tech_names[0]}, {tech_names[1]}'

axs = gm.solutions_2D_small(techs, solutions,
                       # axs = axs,
                      chosen_techs = chosen_techs,
                      cheb = True, opt_system = opt,
                      show_minmax = False,
                      n_samples = n_samples,
                      legend_v = legend_v, legend_h = legend_h,
                      figsize = (20,4),
                      title = title,
                      hist_bins = bins,
                      )

fig = axs[0].get_figure()
filename = f'v_{year}_{study}_{chosen_techs[0]}-{chosen_techs[1]}_MAA_small.pdf'
fig.savefig(f'graphics/{filename}', format = 'pdf', bbox_inches='tight')





# Solutions 2D small for Data/Storage in Local 2040 -----------------------------

studyno = 2
solutions = case_list[studyno]
techs     = techs_list[studyno]
opt       = opt_list[studyno]

# Convert to GW
opt = [x/1000 for x in opt]
solutions = solutions/1000

chosen_techs = ['Data', 'Storage']
year  = 2040
study = 'local'
title = f'{year} - MAA and histogram plot for MAA variables: {tech_names[0]}, {tech_names[1]}'

axs = gm.solutions_2D_small(techs, solutions,
                       # axs = axs,
                      chosen_techs = chosen_techs,
                      cheb = True, opt_system = opt,
                      show_minmax = False,
                      n_samples = n_samples,
                      legend_v = legend_v, legend_h = legend_h,
                      figsize = (20,4),
                      title = title,
                      hist_bins = bins,
                      )

fig = axs[0].get_figure()
filename = f'v_{year}_{study}_{chosen_techs[0]}-{chosen_techs[1]}_MAA_small.pdf'
fig.savefig(f'graphics/{filename}', format = 'pdf', bbox_inches='tight')

#%% MAA densityplot

studyno   = 4
solutions = case_list[studyno]
techs     = techs_list[studyno]
opt       = opt_list[studyno]

filename = '2030_G1_DE-NL_MAA_density.pdf'

# chosen_techs = ['P2X', 'Data']
chosen_techs = ['DE', 'NL']

ax = gm.MAA_density_for_vars(techs, solutions, chosen_techs, n_samples = 1_000_000,
                        opt_system = opt, 
                        legend_v = 0.5, legend_h = 1.6,  ncols = 2,
                        loc = 'center',
                        cheb = True, show_minmax = True,
                        minmax_legend = True,
                        figsize = (7,7))

fig = ax.get_figure()

fig.savefig(f'graphics/{filename}', format = 'pdf', bbox_inches='tight')

# studyno = 2
# solutions = case_list[studyno]
# techs     = techs_list[studyno]
# opt       = opt_list[studyno]

# gm.MAA_density_for_vars(techs, solutions, chosen_techs, n_samples = 10_000,
#                         opt_system = opt, legend_down = -0.3, ncols = 3,
#                         cheb = True, show_minmax = True,
#                         ax = ax)

#%% Chebyshev center and radius - plus reduced chebyshev center

studyno = 4 
solutions = case_list[studyno]
techs     = techs_list[studyno]
opt       = opt_list[studyno]

# Convert to GW
opt = [x/1000 for x in opt]
solutions = solutions/1000

p1 = polytope.qhull(solutions)

cheb_center = p1.chebXc # Chebyshev ball center
cheb_radius = p1.chebR

axs = gm.solutions_2D_small(techs, solutions, chosen_techs = ['DK', 'NL'],
                            n_samples = 100, title = '2040 Links G2',
                            cheb = True, legend_v = -0.3
                            # xlim = [0, None], ylim = [0, None]
                            )

# axs[0]

#
p2 = polytope.qhull(solutions[:,[0,2,3]])

cheb_center = p2.chebXc # Chebyshev ball center

axs[0].plot(cheb_center[0], cheb_center[2],
              marker = 'o', linestyle = '',
              ms = 15, zorder = 3,
              color = 'gold',)

#%% Polyhedron union area

sol1 = np.load('../v_2030_links_G3/v_2030_links_G3_3MAA_10p_solutions.npy')
sol2 = np.load('../v_2040_links_G3/v_2040_links_G3_3MAA_10p_solutions.npy')
techs = ['DK', 'NO', 'BE']

sol11 = sol1.copy()

sol11[:,0] += 100

xlim = [0, sol1[:,0].max()]
ylim = [0, sol1[:,1].max()]
zlim = [0, sol1[:,2].max()]

gm.solutions_3D(techs, sol1, markersize = 2, linewidth = 2,
                xlim = xlim, ylim = ylim, zlim = zlim)
gm.solutions_3D(techs, sol11, markersize = 2, linewidth = 2,
                xlim = xlim, ylim = ylim, zlim = zlim)

intersection = gm.get_intersection(sol1, sol11)

#%% Intersection plot
# gm.solutions_3D(techs, intersection, markersize = 2, linewidth = 2,
#                 xlim = xlim, ylim = ylim, zlim = zlim)
        
gm.plot_intersection(sol1, sol11, intersection)




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
    
    filename = f'graphics/v_{years[i]}_{projects[i]}_{len(techs_list[i])}MAA_10p_plot_2D_MAA_minmax.pdf'
    
    tech_titles = ['IT' if item == 'Data' else item for item in techs_list[i]]
    
    opt = [x/1000 for x in opt_list[i]]
    
    limits = [0, case_list[i].max()/1000]
    
    ncols = 5 if len(techs_list[i]) == 4 else 4
    
    gm.solutions_2D(techs_list[i], case_list[i]/1000,
                    tech_titles = tech_titles,
                    # minmax_techs = ['P2X', 'Data'],
                    n_samples = 1000_000, ncols = ncols,
                    filename = filename,
                    # xlim = limits, ylim = limits,
                    title = titles[i],
                    opt_system = opt,
                    cheb = True, 
                        # show_cheb_radius = True,
                    show_minmax = True,
                    )
    
gm.its_britney_bitch()
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

n_samples   = 10_000
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

projects = ['links_G1']
techs_list = [['DK', 'DE', 'NL', 'BE']]

i = 0
for techs, project in zip(techs_list, projects):
    #Local:
    # filename = f'MAA_overlap_hist_{project}.pdf'
    n_MAA   = len(techs)
    sol30 = np.load(f'../v_2030_{project}/v_2030_{project}_{n_MAA}MAA_10p_solutions.npy')
    sol40 = np.load(f'../v_2040_{project}/v_2040_{project}_{n_MAA}MAA_10p_solutions.npy')
    sols  = [sol30/1000, sol40/1000]
    title = titles[i]
    
    if i == 2:
        y = 0.933
    else:
        y = 0.96
    
    gm.histograms_3MAA(techs, sols, title = title, n_samples = n_samples,
                       titlesize = 20, titley = y,
                       # filename = filename,
                       )
    
    i += 1
    
gm.its_britney_bitch()
    
#%% Anders' plot
import seaborn as sns



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

fig, axs = plt.subplots(6, 1, figsize = (15, 4*6))
fig.subplots_adjust(hspace = 0.5)

colors = gm.get_color_codes()

info = pd.DataFrame(techs_list).T
info.columns = projects

n_samples = 100_000
titlesize = 24
filename  = 'link_specific_histograms.pdf'

countries = tm.get_bus_df()['Abbreviation'][1:].values

pr_name = {'links_G1':'Planned links', 
           'links_G2':'Low-capacity links',
           'links_G3':'High-capacity links'}

i = 0
for variable in countries:
    # variable  = 'BE'
    hatch_project = 'G1'
    
    ax = axs[i]
    
    var_projects = info.loc[:, info.eq(variable).any()]
    
    data_dict = {}  # Initialize an empty dictionary
    
    for project_name in var_projects:
        
        # Remove None values
        var_project = var_projects[project_name].dropna()
        
        for year in [2030, 2040]:
            
            hatch = '/' if hatch_project in project_name else '.'
            label = f'Study: {pr_name[project_name]}' if year == 2030 else '_nolegend_'
            
            n_MAA = len(var_project)
            
            sol = np.load(f'../v_{year}_{project_name}/v_{year}_{project_name}_{n_MAA}MAA_10p_solutions.npy')
            project_list = techs_list[projects.index(project_name)]
            data_dict[(project_name, year)] = {'data': sol, 'list': project_list}
            
            samples = gm.sample_in_hull(sol, n_samples)
            samples_df = pd.DataFrame(samples, columns = project_list) 
            
            linestyle = '-' if year == 2030 else '--'
            dash_pattern = [10, 0] if year == 2030 else [10,10]
            
            sns.histplot(samples_df[variable].values, 
                         line_kws = {'linewidth':3, 'linestyle':linestyle, 
                                     'dashes': dash_pattern},
                         element = 'step',
                         bins = 50,
                         color = colors[variable],
                         alpha = 1/len(var_projects),
                         kde = True,
                         ax = ax, label = label,
                         hatch = hatch,
                         )
            
            if project_name == var_projects.keys()[0]:
                
                line_label = '2030' if linestyle == '-' else '2040'
                ax.plot([], [], linestyle = linestyle, color = colors[variable],
                       label = line_label)
            
    xUnit  = '[MWh]' if variable == 'Storage' else '[MW]'
        
    ax.set(xlabel = f'Installed capacity {xUnit}', 
           ylabel = 'Frequency',
           )
        
    title_var = 'IT' if variable == 'Data' else variable
        
    title_projects = ' '
    for project in var_projects:
        title_projects += pr_name[project] + ', '
        
    axtitle = f'{title_var}' #- from projects: {title_projects}'
        
    ax.set_title(axtitle, color = colors[variable], fontsize = titlesize, y = 0.975)
    ax.set(xlim = [0, None])    
    
    handles, labels = ax.get_legend_handles_labels()
    
    range_list = [a for a in range(2+len(var_projects.keys()))]
    # Reorder the legend entries
    order = range_list[1:] + [range_list[0]]
    # order = [range_list[-1]] + range_list[:-1]
    handles = [handles[r] for r in order]
    labels = [labels[r] for r in order]
    
    ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.35, 1))
            
    i += 1
    
fig.savefig('graphics/'+filename, format = 'pdf', bbox_inches='tight')

gm.its_britney_bitch()
    
#%% Chebychev

project = 'local'
n_MAA   = 3
sol     = np.load(f'../v_2030_{project}/v_2030_{project}_{n_MAA}MAA_10p_solutions.npy')
    

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

#%% MAA density + Histogram v2

solutions = sol1

gm.solutions_2D_small(techs, solutions, chosen_techs = ['P2X', 'Data'])

#%% Histogram + MAA density function
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull
from gorm import sample_in_hull

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


#%% local_nac case MAA near-optimal space diagrams

studyno = 3 
year    = 2040
solutions = case_list[studyno]
techs     = techs_list[studyno]
opt       = opt_list[studyno]

# Convert to GW
opt = [x/1000 for x in opt]
solutions = solutions/1000

n_samples = 1_000_000

# Initialize figure
plt.figure()

fig, axs = plt.subplots(1, 3, figsize = (18,5))
fig.subplots_adjust(wspace = 0.3, hspace = 0.3)
fig.suptitle(f'{year} Unconstrained area MAA - near optimal spaces', fontsize = 24)

# chosen_techs = ['P2X', 'Data']
# year  = 2030
# study = 'G2'

gm.MAA_density_for_vars(techs, solutions, ['P2X', 'Data'],
                        n_samples = n_samples,
                        ax = axs[0],
                        opt_system = opt, cheb = True,
                        show_legend = False,
                        )

gm.MAA_density_for_vars(techs, solutions, ['P2X', 'Storage'],
                        n_samples = n_samples,
                        ax = axs[1],
                        opt_system = opt, cheb = True,
                        show_legend = True, ncols = 4,
                        legend_v = -0.3
                        )

gm.MAA_density_for_vars(techs, solutions, ['Data', 'Storage'],
                        n_samples = n_samples,
                        ax = axs[2],
                        opt_system = opt, cheb = True,
                        show_legend = False,
                        )

# title = f'{year} - MAA and histogram plot for MAA variables: {chosen_techs[0]}, {chosen_techs[1]}'

fig = axs[0].get_figure()

filename = f'MAA_{year}_unconstrained_spaces.pdf'

fig.savefig(f'graphics/{filename}', format = 'pdf', bbox_inches='tight')

#%% 3D plots

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

# for i in [8]: #range(len(case_list)):
    
filename = f'graphics/3D/v_{years[i]}_{projects[i]}_3D.pdf'

i = 6

techs     = techs_list[i]
solutions = case_list[i]/1000
opt       = opt_list[i]

tech_titles = ['IT' if item == 'Data' else item for item in techs]

opt = [x/1000 for x in opt]

ax = gm.solutions_3D(techs, solutions,
                title = '3D - near-optimal space for \n High capacity links, 2030 vs 2040',
                markersize = 0, linewidth = 0.5,
                # filename = filename,
                )

i = 9

techs     = techs_list[i]
solutions = case_list[i]/1000
opt       = opt_list[i]

tech_titles = ['IT' if item == 'Data' else item for item in techs]

opt = [x/1000 for x in opt]

gm.solutions_3D(techs, solutions, ax = ax,
                title = f'3D - {titles[i]} - i: {i}',
                markersize = 0, linewidth = 0.5,
                # filename = filename,
                )
    
# fig = ax.get_figure()
    
# fig.savefig('graphics/3D/G3_2030_vs_2040.pdf', format = 'pdf', bbox_inches='tight')


    # ncols = 5 if len(techs_list[i]) == 4 else 4
    
    
    # gm.solutions_2D(techs_list[i], case_list[i]/1000,
    #                 tech_titles = tech_titles,
    #                 # minmax_techs = ['P2X', 'Data'],
    #                 n_samples = 1000_000, ncols = ncols,
    #                 # filename = filename,
    #                 # xlim = limits, ylim = limits,
    #                 title = titles[i],
    #                 opt_system = opt,
    #                 cheb = True, 
    #                     # show_cheb_radius = True,
    #                 show_minmax = True,
    #                 )

#%% Export 3D Models
# 3D cases: 0, 1, 2, 3,   5, 6,   8, 9

import numpy as np
from scipy.spatial import ConvexHull

for case_number in [0, 1, 2, 3, 5, 6, 8, 9]:
    solutions = case_list[case_number]
    
    gm.generate_3D_model(solutions, f'3D_models/output_{case_number}.stl')

#%% 

i = 6

techs     = techs_list[i]
solutions = case_list[i]/1000
opt       = opt_list[i]

tech_titles = ['IT' if item == 'Data' else item for item in techs]

opt = [x/1000 for x in opt]

ax = gm.solutions_3D(techs, solutions,
                title = f'3D',
                opt = opt,
                markersize = 0, linewidth = 0.5,
                # filename = filename,
                )






















