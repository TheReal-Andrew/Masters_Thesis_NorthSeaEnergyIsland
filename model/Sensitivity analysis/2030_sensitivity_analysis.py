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
year       = 2030
input_name = '../Base0/v_' + str(year) +'_base0_opt.nc'

#%% Set up network and load in data
n = pypsa.Network(input_name) #Load network from netcdf file

# ----- data ---------
n.area_use      = tm.get_area_use()
n.total_area    = tm.get_main_parameters()[0][year]["island_area"]
n.link_sum_max  = n.generators.p_nom_max['Wind']
n.main_links    = n.links[~n.links.index.str.contains("bus")].index

n_opt = n.copy()

#%% Define custom optimization constraints
def extra_functionality(n,snapshots):
    gm.area_constraint(n, snapshots)
    gm.link_constraint(n, snapshots)

#%% Should run or not
cc_mode = False  # Choose to run capital cost sweep or not
mc_mode = True # Choose to run marginal cost sweep or not
    
#%% Set sensitivity sweep parameters

cc_ranges = {
    'P2X': np.arange(0,4.05,0.05),
    'Data': np.arange(0,4.05,0.05),
    'Island_store': np.arange(0,4.05,0.05),
    'Links': np.arange(0,4.05,0.05),    
    }

mc_ranges = {
    'P2X': np.arange(0,4.05,0.05),
    'Data': np.arange(0,4.05,0.05),
    'Island_store': np.arange(0,4.05,0.05),
    'Country_gen': np.arange(0,4.05,0.05),
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
            
            n = n_opt.copy()
            
            # ----- data ---------
            n.area_use      = tm.get_area_use()
            n.total_area    = tm.get_main_parameters()[0][year]['island_area']
            n.link_sum_max  = n_opt.generators.p_nom_max['Wind']
            n.main_links    = n_opt.links[~n_opt.links.index.str.contains("bus")].index
            
            # Change capital cost
            if component == 'Island_store':
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
            cc_sensitivity_cap[component]['Island_store'].append(n.stores.loc['Island_store'].e_nom_opt)
            
            # Store optimum system price
            cc_sensitivity_cap[component]['Optimum'].append(n.objective)
        
            # Update count of the number of studies done
            current_cc_n_study = current_cc_n_study + 1
            
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
            
            n = n_opt.copy()
            
            # ----- data ---------
            n.area_use      = tm.get_area_use()
            n.total_area    = tm.get_main_parameters()[0][year]['island_area']
            n.link_sum_max  = n_opt.generators.p_nom_max['Wind']
            n.main_links    = n_opt.links[~n_opt.links.index.str.contains("bus")].index
            
            # Change marginal cost
            if component == 'Island_store':
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
            mc_sensitivity_cap[component]['Island_store'].append(n.stores.loc['Island_store'].e_nom_opt)
            
            # Store optimum system price
            mc_sensitivity_cap[component]['Optimum'].append(n.objective)
            
            # Update count of the number of studies done
            current_mc_n_study = current_mc_n_study + 1
            
    # save dictionary to person_data.pkl file
    with open(str(year) +'_mc_sensitivity_cap.pkl', 'wb') as fp:
        pickle.dump(mc_sensitivity_cap, fp)

else:
    # Read data from file
    with open(str(year) + '_mc_sensitivity_cap.pkl', 'rb') as fp:
        mc_sensitivity_cap = pickle.load(fp)
        
t.toc()
gm.its_britney_bitch()  

#%% 


      
#%% Plot sweep: generators + storage

plot_components = []

for i in cc_components:
    if i in mc_components and i not in plot_components:
        plot_components.append(i)

for component in plot_components:
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7*2,5), dpi = 300, constrained_layout = True)
    for i in range(2):
        if i == 0:
            x1 = cc_sensitivity_cap[component]['Step'].copy()
            
            axs[i].set_title(f'{year}: Sensitivity of\n{component} capital cost', pad = 5)
            axs[i].set_xlabel('Capital cost coefficient [-]')
            axs[i].set_ylabel('Area use [m2]')
            
            # axs[i].set_ylim([-100,100])
            # axs[i].set_yticks(np.arange(0,1.1,0.1))
            
            axs_copy = axs[i].twinx()
            axs_copy.get_yaxis().set_visible(False)
            
            for k in [s for s in (list(n.generators.index) + list(n.stores.index)) if s in cc_components]:
                y1 = cc_sensitivity_cap[component][k].copy()
                
                if k == "Island_store":
                    y1 = [i * n.area_use['storage'] for i in y1]
                    y1 = [i / n.total_area for i in y1]
                    y1 = [i * 100 for i in y1]
                elif k == "P2X":
                    y1 = [q * n.area_use['hydrogen'] for q in y1]
                    y1 = [w / n.total_area for w in y1]
                    y1 = [e * 100 for e in y1]

                elif k == "data":
                    y1 = [i * n.area_use['data'] for i in y1]
                    y1 = [i / n.total_area for i in y1]
                    y1 = [i * 100 for i in y1]
                    
                print(y1)
                        
                axs[i].plot(x1, y1, linestyle='-', marker='.', label = k, linewidth = 3)
                
            cc_optimum = cc_sensitivity_cap[component]['Optimum'].copy()
            cc_optimum[:] = [x / max(cc_optimum) for x in cc_optimum]
            
            axs_copy.plot(x1, cc_optimum, linestyle='-.', marker='.', color = 'k', label = 'Optimum', linewidth = 0.9)
            axs_copy.set_ylabel('Objective optimum [€]') 
            axs_copy.set_ylim([-0.05,1.05])   
            
            # if component == 'Data':
            #     axs[i].set_xlim([0,4])
            #     axs[i].set_xticks(np.arange(0,4.5,0.5))   
            # if component == 'P2X':
            #     axs[i].set_xlim([0,4])
            #     axs[i].set_xticks(np.arange(0,4.5,0.5))
            # if component == 'Island_store':
            #     axs[i].set_xlim([0,4])
            #     axs[i].set_xticks(np.arange(0,4.5,0.5))
            
        if i == 1:
            x2 = mc_sensitivity_cap[component]['Step'].copy()
            
            axs[i].set_title(f'{year}: Sensitivity analysys of\n{component} marginal revenue', pad = 10)
            
            # axs[i].set_ylim([-100,100])
            # axs[i].set_yticks(np.arange(0,1.1,0.1))
            axs[i].set_yticklabels([])
            axs_copy = axs[i].twinx()
            axs_copy.set_yticks(np.arange(0,1.1,0.1))
                
            for k in [s for s in (list(n.generators.index) + list(n.stores.index)) if s in mc_components]:
                y2 = mc_sensitivity_cap[component][k].copy()
                
                # if k == "Island_store":
                #     for j in range(len(y2)):
                #         try:
                #             y2[j] = (y2[j] / n_opt.stores.e_nom_opt[k] - 1)*100
                #         except ZeroDivisionError:
                #             y2[j] = 0
                # else:
                #     for j in range(len(y2)):
                #         try:
                #             y2[j] = (y2[j] / n_opt.generators.p_nom_opt[k] - 1)*100
                #         except ZeroDivisionError:
                #             y2[j] = 0
                
                if k == "Island_store":
                    y2 = [i * n.area_use['storage'] for i in y2]
                    y2 = [i / n.total_area for i in y2]
                    y2 = [i * 100 for i in y2]
                elif k == "P2X":
                    y2 = [i * n.area_use['hydrogen'] for i in y2]
                    y2 = [i / n.total_area for i in y2]
                    y2 = [i * 100 for i in y2]
                elif k == "data":
                    y2 = [i * n.area_use['data'] for i in y2]
                    y2 = [i / n.total_area for i in y2]
                    y2 = [i * 100 for i in y2]
                        
                axs[i].plot(x2, y2, linestyle='-', marker='.', label = k, linewidth = 3)
            
            mc_optimum = mc_sensitivity_cap[component]['Optimum'].copy()
            mc_optimum[:] = [x / max(mc_optimum) for x in mc_optimum]
            
            axs_copy.plot(x2, mc_optimum, linestyle='-.', marker='.', color = 'k', label = 'Optimum', linewidth = 0.9)
            axs_copy.set_ylabel('Norm. objective optimum [-]') 
            axs_copy.set_ylim([-0.05,1.05])    
            
            if component == 'Data':
                axs[i].set_xlim([0,4])
                axs[i].set_xticks(np.arange(0,4.5,0.5))
                axs[i].set_xlabel('Marginal revenue coefficient [-]')
            if component == 'P2X':
                axs[i].set_xlim([0,4])
                axs[i].set_xticks(np.arange(0,4.5,0.5))
                axs[i].set_xlabel('Marginal revenue coefficient [-]')
            if component == 'Island_store':
                axs[i].set_xlim([0,4])
                axs[i].set_xticks(np.arange(0,4.5,0.5))
                axs[i].set_xlabel('Marginal cost coefficient [-]')
                axs[i].set_title(f'{year}: Sensitivity analysys of\n{component} marginal cost', pad = 10)
            
            lines, labels = axs[i].get_legend_handles_labels()
            lines2, labels2 = axs_copy.get_legend_handles_labels()
            axs[i].legend(lines + lines2, labels + labels2, loc='center right', fontsize = 15)    
            
    # plt.tight_layout() 
    plt.savefig('../../images/sensitivity/' + str(year) + '_' + component + '_sensitivity.pdf', format = 'pdf', bbox_inches='tight')
plt.show()

#%% Plot sweep: links
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7,5), dpi = 300, constrained_layout = True)

for i in n.main_links:
    try:
        axs.plot(cc_sensitivity_cap['Links'][i], label = i)
        axs.legend(loc = 'best')
    except KeyError:
        plt.clf()
        continue
    
#%% Plot sweep: country generators
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7,5), dpi = 300, constrained_layout = True)
for i in n.generators.index:
    if i in ['Wind','MoneyBin','P2X','Data']:
        continue
    try:
        axs.plot(mc_sensitivity_cap['Country_gen'][i], label = i)
        axs.legend(loc = 'best')
    except KeyError:
        plt.clf()
        continue