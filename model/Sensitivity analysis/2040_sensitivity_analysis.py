#%% Import packages
import os
import sys
os.chdir(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.abspath('../../modules'))
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pypsa
import gorm as gm
import tim as tm
import island_plt as ip
ip.set_plot_options()
# import pandas as pd
from matplotlib.ticker import (MultipleLocator)
from pytictoc import TicToc
t = TicToc() #create instance of time class

#%% Control

year       = 2040
input_name = '../Base0/v_' + str(year) +'_base0_opt.nc'

connected_countries =  [
                        "Denmark",         
                        "Norway",          
                        "Germany",         
                        "Netherlands",     
                        "Belgium",         
                        "United Kingdom"
                        ]

#%Set up network and load in data
n = pypsa.Network(input_name) #Load network from netcdf file

# ----- data ---------
n.area_use              = tm.get_area_use()
n.total_area            = tm.get_main_parameters()[0][year]['island_area']
n.link_sum_max          = n.generators.p_nom_max['Wind']
n.main_links            = n.links.loc[n.links.bus0 == "Energy Island"].index
n.connected_countries   = connected_countries

n_opt = n.copy()
n_opt.area_use          = tm.get_area_use()
n_opt.area_use.rename(index={'storage':'Storage'})
n_opt.area_use.rename(index={'hydrogen':'P2X'})
n_opt.area_use.rename(index={'data':'Data'})

# Define custom optimization constraints
def extra_functionality(n,snapshots):
    gm.area_constraint(n, snapshots)
    gm.link_constraint(n, snapshots)
    gm.marry_links(n, snapshots)

n.lopf(pyomo                = False,
       solver_name          = 'gurobi',
       keep_shadowprices    = True,
       keep_references      = True,
       extra_functionality  = extra_functionality)
#%% Define custom optimization constraints
def extra_functionality(n,snapshots):
    gm.area_constraint(n, snapshots)
    gm.link_constraint(n, snapshots)
    gm.marry_links(n, snapshots)

#%% Should run or not
cc_mode = True # Choose to run capital cost sweep or not
mc_mode = True # Choose to run marginal cost sweep or not
    
#%% Set sensitivity sweep parameters

cc_ranges = {
    'P2X': np.arange(0,1.05,0.05),
    'Data': np.arange(1,4.05,0.05),
    'Storage': np.arange(0,1.05,0.05),
    # 'Links': np.arange(0,4.05,0.05),    
    }

mc_ranges = {
    'P2X': np.arange(1,4.05,0.05),
    'Data': np.arange(0,1.05,0.05),
    'Storage': np.arange(0,1.05,0.05),
    # 'Country_gen': np.arange(0,4.05,0.05),
    }
    
cc_n_studies = 0
mc_n_studies = 0  
for item in cc_ranges:
    cc_n_studies = len(cc_ranges[item]) + cc_n_studies
for item in mc_ranges:
    mc_n_studies = len(mc_ranges[item]) + mc_n_studies
    
print('Number of CC studies: ' + str(cc_n_studies))
print('Number of MC studies: ' + str(mc_n_studies))
#%% Sensitivity sweep: capital cost
t.tic()
cc_components = list(cc_ranges.keys())
if cc_mode == True:
    current_cc_n_study = 1
    
    # Initialize/reset dictionaries
    cc_sensitivity_cap = {key: None for key in cc_components}
    for component in cc_components:
        cc_sensitivity_cap[component] = {key: [] for key in (list(n.generators.index) + list(n.stores.index) + list(n.main_links) + ['Optimum'] + ['Step'])}
        cc_sensitivity_cap[component]['Step'] = cc_ranges[component]
    
    for i, component in enumerate(cc_components):
        
        for cc in cc_ranges[component]:
            
            print('')
            print('########################################')
            print('CC study number: ' + str(current_cc_n_study) + ' out of ' + str(cc_n_studies))
            print('########################################')
            print('')
            
            t_cc_loop = TicToc() #create instance of time class
            t_cc_loop.tic()
            
            n = n_opt.copy()
            
            # ----- data ---------
            n.area_use              = tm.get_area_use()           
            n.total_area            = tm.get_main_parameters()[0][year]['island_area']
            n.link_sum_max          = n.generators.p_nom_max['Wind']
            n.main_links            = n.links.loc[n.links.bus0 == "Energy Island"].index
            n.connected_countries   = connected_countries
            
            # Change capital cost
            if component == 'Storage':
                n.stores.loc[component, 'capital_cost'] = n_opt.stores.loc[component, 'capital_cost'] * cc
            elif component == 'Links':
                for l in n.main_links:
                    n.links.loc[l, 'capital_cost'] = n_opt.links.loc[l, 'capital_cost'] * cc
            else:
                n.generators.loc[component, 'capital_cost'] = n_opt.generators.loc[component, 'capital_cost'] * cc
                
            # Solve
            n.lopf(pyomo                = False,
                   solver_name          = 'gurobi',
                   keep_shadowprices    = True,
                   keep_references      = True,
                   extra_functionality  = extra_functionality)
            
            # Store installed tech. capacities
            for j in n.generators.index:
                cc_sensitivity_cap[component][j].append(n.generators.p_nom_opt[j])
            
            # Store installed link capacities
            for k in n.main_links:
                cc_sensitivity_cap[component][k].append(n.links.loc[k].p_nom_opt)
            
            # Store installed storage capacity    
            cc_sensitivity_cap[component]['Storage'].append(n.stores.loc['Storage'].e_nom_opt)
            
            # Store optimum system price
            cc_sensitivity_cap[component]['Optimum'].append(n.objective)
            
            # Update count of the number of studies done
            current_cc_n_study = current_cc_n_study + 1
            
            print('')
            print('########################################')
            print('Estimated time left of CC study = ' + str(t_cc_loop.tocvalue()/60 * (cc_n_studies - current_cc_n_study)))
            print('########################################')
            print('')
            
        # save dictionary to person_data.pkl file
    with open(str(year) +'_cc_sensitivity_cap.pkl', 'wb') as fp:
        pickle.dump(cc_sensitivity_cap, fp)

else:
    # Read dictionary pkl file
    with open(str(year) +'_cc_sensitivity_cap.pkl', 'rb') as fp:
        cc_sensitivity_cap = pickle.load(fp)
            
#%% Sensitivity sweep: marginal cost
mc_components = list(mc_ranges.keys())
if mc_mode == True:
    current_mc_n_study = 1
    
    # Initialize/reset dictionaries
    mc_sensitivity_cap = {key: None for key in mc_components}
    for component in mc_components:
        mc_sensitivity_cap[component] = {key: [] for key in (list(n.generators.index) + list(n.stores.index) + list(n.main_links) + ['Optimum'] + ['Step'])}
        mc_sensitivity_cap[component]['Step'] = mc_ranges[component]
    
    for i, component in enumerate(mc_components):
        
        for mc in mc_ranges[component]:
            
            print('')
            print('########################################')
            print('MC study number: ' + str(current_mc_n_study) + ' out of ' + str(mc_n_studies))
            print('########################################')
            print('')
            
            t_mc_loop = TicToc() #create instance of time class
            t_mc_loop.tic()
            
            n = n_opt.copy()
            
            # ----- data ---------
            n.area_use              = tm.get_area_use()           
            n.total_area            = tm.get_main_parameters()[0][year]['island_area']
            n.link_sum_max          = n.generators.p_nom_max['Wind']
            n.main_links            = n.links.loc[n.links.bus0 == "Energy Island"].index
            n.connected_countries   = connected_countries
            
            # Change marginal cost
            if component == 'Storage':
                n.stores.loc[component, 'marginal_cost'] = n_opt.stores.loc[component, 'marginal_cost'] * mc
            elif component == 'Country_gen':
                for l in n.generators.index:
                    if l in ['Wind','MoneyBin','P2X','Data']:
                        continue
                    n.generators_t['marginal_cost'][l] = n_opt.generators_t['marginal_cost'][l] * mc
            else:
                n.generators.loc[component, 'marginal_cost'] = n_opt.generators.loc[component, 'marginal_cost'] * mc
    
            # Solve
            n.lopf(pyomo                = False,
                   solver_name          = 'gurobi',
                   keep_shadowprices    = True,
                   keep_references      = True,
                   extra_functionality  = extra_functionality)
            
            # Store installed tech. capacities
            for j in n.generators.index:
                mc_sensitivity_cap[component][j].append(n.generators.p_nom_opt[j])
            
            # Store installed link capacities
            for k in n.main_links:
                mc_sensitivity_cap[component][k].append(n.links.loc[k].p_nom_opt)
            
            # Store installed storage capacity    
            mc_sensitivity_cap[component]['Storage'].append(n.stores.loc['Storage'].e_nom_opt)
            
            # Store optimum system price
            mc_sensitivity_cap[component]['Optimum'].append(n.objective)
            
            # Update count of the number of studies done
            current_mc_n_study = current_mc_n_study + 1
            
            print('')
            print('########################################')
            print('Estimated time left of MC study = ' + str(t_mc_loop.tocvalue()/60 * (mc_n_studies - current_mc_n_study)))
            print('########################################')
            print('')
            
    # save dictionary to person_data.pkl file
    with open(str(year) +'_mc_sensitivity_cap.pkl', 'wb') as fp:
        pickle.dump(mc_sensitivity_cap, fp)

else:
    # Read data from file
    with open(str(year) + '_mc_sensitivity_cap.pkl', 'rb') as fp:
        mc_sensitivity_cap = pickle.load(fp)
        
t.toc()
# gm.its_britney_bitch()        
#%% Plot sweep: generators + storage
colors = gm.get_color_codes()
plot_components = []

for i in cc_components:
    if i in mc_components and i not in plot_components:
        plot_components.append(i)

for component in plot_components:
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7*2,5), dpi = 300, constrained_layout = True)
    for i in range(2):
        if i == 0:

            axs[i].set_title(f'{year}: Sensitivity of\n{component} capital cost', pad = 5)
            axs[i].set_xlabel('Capital cost coefficient [-]')
            axs[i].set_ylabel('Use of available island area [%]')
            
            axs[i].set_ylim([-0.05,1.05])
            
            axs[i].set_yticks(np.arange(0,1.2,0.2),[0,20,40,60,80,100])
            axs[i].yaxis.set_minor_locator(MultipleLocator(0.05))
            axs[i].xaxis.set_minor_locator(MultipleLocator(0.1))
            
            axs_copy = axs[i].twinx()
            axs_copy.get_yaxis().set_visible(False)
            
            for k in [s for s in (list(n.generators.index) + list(n.stores.index)) if s in cc_components]:
                y1 = cc_sensitivity_cap[component][k].copy()
                
                if k == "Data":
                    for j in range(len(y1)):
                        y1[j] = y1[j]  * n.area_use['data'] / n.total_area
                elif k == "P2X":
                    for j in range(len(y1)):
                        y1[j] = y1[j] * n.area_use['hydrogen'] / n.total_area
                elif k == "Storage":
                    for j in range(len(y1)):
                        y1[j] = y1[j] * n.area_use['storage'] / n.total_area    
                
                x1 = cc_sensitivity_cap[component]['Step'].copy()
                axs[i].plot(x1, y1, linestyle='-', marker='.', label = k, linewidth = 3, color = colors[k])
                
            cc_optimum = cc_sensitivity_cap[component]['Optimum'].copy()
            cc_optimum[:] = [x / max(cc_optimum) for x in cc_optimum]
            
            axs_copy.plot(x1, cc_optimum, linestyle='-.', marker='.', markersize=4, color = 'k', label = 'Optimum', linewidth = 0.4)
            axs_copy.set_ylabel('Objective optimum [€]') 
            axs_copy.set_ylim([-0.05,1.05])
            
            if component == 'Data':
                axs[i].set_xlim([1,4])
                axs[i].set_xticks(np.arange(1,4.5,0.5))   
            if component == 'P2X':
                axs[i].set_xlim([0,1])
                axs[i].set_xticks(np.arange(0,1.1,0.1))
            if component == 'Storage':
                axs[i].set_xlim([0,1])
                axs[i].set_xticks(np.arange(0,1.1,0.1))
            
        if i == 1:
            
            axs[i].set_title(f'{year}: Sensitivity analysys of\n{component} marginal revenue', pad = 10)
            
            axs[i].set_ylim([-0.05,1.05])
            axs[i].set_yticks(np.arange(0,1.2,0.2))
            axs[i].set_yticklabels([])
            
            for k in [s for s in (list(n.generators.index) + list(n.stores.index)) if s in mc_components]:
                y2 = mc_sensitivity_cap[component][k].copy()
                print(k)
                if k == "Data":
                    for j in range(len(y2)):
                        y2[j] = y2[j] * n.area_use['data'] / n.total_area
                elif k == "P2X":
                    for j in range(len(y2)):
                        y2[j] = y2[j] * n.area_use['hydrogen'] / n.total_area
                elif k == "Storage":
                    for j in range(len(y2)):
                        y2[j] = y2[j] * n.area_use['storage'] / n.total_area 
                        
                x2 = mc_sensitivity_cap[component]['Step'].copy()        
                axs[i].plot(x2, y2, linestyle='-', marker='.', label = k, linewidth = 3, color = colors[k])
            
            axs[1].yaxis.set_minor_locator(MultipleLocator(0.05))
            axs[1].xaxis.set_minor_locator(MultipleLocator(0.1))
            
            axs_copy2 = axs[1].twinx()
            axs_copy2.set_ylim([-0.05,1.05])
            axs_copy2.set_ylabel('Normalized optimum [-]')
            axs_copy2.yaxis.set_minor_locator(MultipleLocator(0.05))
            
            mc_optimum = mc_sensitivity_cap[component]['Optimum'].copy()
            mc_optimum[:] = [x / max(mc_optimum) for x in mc_optimum]
            
            axs_copy2.plot(x2, mc_optimum, linestyle='-.', marker='.', markersize=4, color = 'k', label = 'Optimum', linewidth = 0.4)
            
            if component == 'Data':
                axs[i].set_xlim([0,1])
                axs[i].set_xticks(np.arange(0,1.1,0.1))
                axs[i].set_xlabel('Marginal revenue coefficient [-]')
            if component == 'P2X':
                axs[i].set_xlim([1,4])
                axs[i].set_xticks(np.arange(1,4.5,0.5))
                axs[i].set_xlabel('Marginal revenue coefficient [-]')
            if component == 'Storage':
                axs[i].set_xlim([0,1])
                axs[i].set_xticks(np.arange(0,1.1,0.1))
                axs[i].set_xlabel('Marginal cost coefficient [-]')
                axs[i].set_title(f'{year}: Sensitivity analysys of\n{component} marginal cost', pad = 10)
            
            lines, labels = axs[i].get_legend_handles_labels()
            lines2, labels2 = axs_copy.get_legend_handles_labels()
            axs[i].legend(lines + lines2, labels + labels2, loc='upper right', bbox_to_anchor=(1.01, 0.96), fontsize = 15)    
            
    # plt.tight_layout() 
    plt.savefig('../../images/sensitivity/' + str(year) + '_' + component + '_sensitivity.pdf', format = 'pdf', bbox_inches='tight')
plt.show()

#%% Plot sweep: links
# fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7,5), dpi = 300, constrained_layout = True)

# for i in n.main_links:
    
#     if i.startswith('Island_to_'):
    
#         # i = i.replace("_"," ")
#         x = cc_sensitivity_cap['Links']['Step'].copy()
#         y = cc_sensitivity_cap['Links'][i]
        
#         axs.plot(x, y, label = i)
#         axs.legend(loc = 'best')
#     else:
#         continue
    
# plt.title(f'{year}: Sensitivity of link capital cost', pad = 5)
# plt.xlabel('Capital cost coefficient [-]')
# plt.ylabel('Installed capacity [MW]')

    
# #%% Plot sweep: country generators
# fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7,5), dpi = 300, constrained_layout = True)
# for i in n.generators.index:
#     if i in ['Wind','MoneyBin','P2X','Data']:
#         continue
#     try:
#         axs.plot(mc_sensitivity_cap['Country_gen'][i], label = i)
#         axs.legend(loc = 'best')
#     except KeyError:
#         plt.clf()
#         continue

# plt.title(f'{year}: Sensitivity of country  capital cost', pad = 5)
# plt.xlabel('Capital cost coefficient [-]')
# plt.ylabel('Installed capacity [MW]')
