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

#%%

# Control
year       = 2030
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

#%% Should run or not
area_mode = False # Choose to run capital cost sweep or not
    
#%% Set sensitivity sweep parameters

area_ranges = {
    'hydrogen': np.arange(0.05,1.05,0.05),
    'data':     np.arange(1,4.1,0.1),
    'storage':  np.arange(0.05,1.05,0.05),
    # 'hydrogen': np.arange(1,3,1),
    # 'data':     np.arange(1,3,1),
    # 'storage':  np.arange(1,3,1),
    }
    
# np.arange(0,4.05,0.05)
area_n_studies = 0

for item in area_ranges:
    area_n_studies = len(area_ranges[item]) + area_n_studies

print('Number of area studies: ' + str(area_n_studies))
#%% Sensitivity sweep: capital cost
t.tic()
area_components = list(area_ranges.keys())

if area_mode == True:
    current_area_n_study = 1
    
    # Initialize/reset dictionaries
    area_sensitivity_cap = {key: None for key in area_components}
    for component in area_components:
        area_sensitivity_cap[component] = {key: [] for key in (list(n.generators.index) + list(n.stores.index) + list(n.main_links) + ['Optimum'] + ['Step'] + ['Use_area'])}
        area_sensitivity_cap[component]['Step'] = area_ranges[component]
    
    for i, component in enumerate(area_components):
        
        for area in area_ranges[component]:
            
            print('')
            print('########################################')
            print('Area study number: ' + str(current_area_n_study) + ' out of ' + str(area_n_studies))
            print('########################################')
            print('')
            
            t_area_loop = TicToc() #create instance of time class
            t_area_loop.tic()
            
            n = n_opt.copy()
            
            # ----- data ---------
            n.area_use              = tm.get_area_use()
            n.area_use.rename(index={'storage':'Storage'})
            n.area_use.rename(index={'hydrogen':'P2X'})
            n.area_use.rename(index={'data':'Data'})
            
            n.total_area            = tm.get_main_parameters()[0][year]['island_area']
            n.link_sum_max          = n.generators.p_nom_max['Wind']
            n.main_links            = n.links.loc[n.links.bus0 == "Energy Island"].index
            n.connected_countries   = connected_countries
            
            # Change area use
            n.area_use.loc[component] = n_opt.area_use.loc[component] * area
           
            # Solve
            n.lopf(pyomo                = False,
                   solver_name          = 'gurobi',
                   keep_shadowprices    = True,
                   keep_references      = True,
                   extra_functionality  = extra_functionality)
            
            # Store installed tech. capacities
            for j in n.generators.index:
                area_sensitivity_cap[component][j].append(n.generators.p_nom_opt[j])
            
            # Store installed link capacities
            for k in n.main_links:
                area_sensitivity_cap[component][k].append(n.links.loc[k].p_nom_opt)
            
            # Store installed storage capacity    
            area_sensitivity_cap[component]['Storage'].append(n.stores.loc['Storage'].e_nom_opt)
            print("Storage = " + str(round(n.stores.loc['Storage'].e_nom_opt)) + ", area_use coef. = " + str(area))
            print(n.area_use)
            
            # Store optimum system price
            area_sensitivity_cap[component]['Optimum'].append(n.objective)
            
            # Save area use
            area_sensitivity_cap[component]['Use_area'].append(n.area_use)
            
            # Update count of the number of studies done
            current_area_n_study = current_area_n_study + 1
            
            print('')
            print('########################################')
            print('Estimated time left of area study = ' + str(t_area_loop.tocvalue()/60 * (area_n_studies - current_area_n_study)))
            print('########################################')
            print('')
            
        # save dictionary to person_data.pkl file
    with open(str(year) +'_area_sensitivity_cap.pkl', 'wb') as fp:
        pickle.dump(area_sensitivity_cap, fp)

else:
    # Read dictionary pkl file
    with open(str(year) +'_area_sensitivity_cap.pkl', 'rb') as fp:
        area_sensitivity_cap = pickle.load(fp)
        
t.toc()
gm.its_britney_bitch()        
#%% Plot sweep: generators + storage
area_components = list(area_ranges.keys())
plot_components = []

titler = ['P2X','IT','storage']

colors = gm.get_color_codes()

for i in area_components:
    plot_components.append(i)

area_components = ['P2X','Data', 'Storage']

for q, component in enumerate(plot_components):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7*2,5), dpi = 300, constrained_layout = True)
    for i in range(2):
        if i == 0:
            x1 = area_sensitivity_cap[component]['Step'].copy()
            titel = titler[q]
            # axs[0].set_title(f'{year}: Sensitivity of\n{component} area use', pad = 5)
            fig.suptitle(f'{year}: Sensitivity of {titel} area use')
            axs[0].set_xlabel('Area use coefficient [-]')
            axs[0].set_ylabel('Use of available island area [%]')
            
            # axs[0].set_xlim([0.5,2])
            # axs[0].set_xticks(np.arange(0.5,2.25,0.25))
            
            if component == "hydrogen":
                axs[0].set_xlim(0,max(area_ranges[component]))
            if component == "data":
                axs[0].set_xlim(min(area_ranges[component]),max(area_ranges[component]))
            if component == "storage":
                axs[0].set_xlim(0,max(area_ranges[component]))
            
            axs[0].set_ylim([-0.05,1.05])
            
            axs[0].set_yticks(np.arange(0,1.2,0.2),[0,20,40,60,80,100])
            axs[0].yaxis.set_minor_locator(MultipleLocator(0.05))
            axs[0].xaxis.set_minor_locator(MultipleLocator(0.1))
            
            axs_copy1 = axs[0].twinx()
            axs_copy1.set_ylim([-0.05,1.05])
            axs_copy1.get_yaxis().set_visible(False)
            
            area_optimum = area_sensitivity_cap[component]['Optimum'].copy()
            area_optimum[:] = [x / max(area_optimum) for x in area_optimum]
            axs_copy1.plot(x1, area_optimum, linestyle='-.', marker='.', markersize=4, color = 'k', label = 'Optimum', linewidth = 0.4)
            
            #Plot area use
            for k in [s for s in (list(n.generators.index) + list(n.stores.index)) if s in area_components]:
                y1 = area_sensitivity_cap[component][k].copy()
                if k == "Data":
                    for j in range(len(y1)):
                        y1[j] = y1[j]  * area_sensitivity_cap.copy()[component]['Use_area'][j]['data'] / n.total_area
                elif k == "P2X":
                    for j in range(len(y1)):
                        y1[j] = y1[j] * area_sensitivity_cap.copy()[component]['Use_area'][j]['hydrogen'] / n.total_area
                elif k == "Storage":
                    for j in range(len(y1)):
                        y1[j] = y1[j] * area_sensitivity_cap.copy()[component]['Use_area'][j]['storage'] / n.total_area                  
                        
                axs[0].plot(x1, y1, linestyle='-', marker='.', label = k, linewidth = 3, color = colors[k])
                
            # plt.legend(loc = 'best')
            
        if i == 1:
            #Plot capacities 
            for k in [s for s in (list(n.generators.index) + list(n.stores.index)) if s in area_components]:
                y2 = area_sensitivity_cap[component][k].copy()
                axs[1].plot(x1, y2, linestyle='-', marker='.', label = k, linewidth = 3, color = colors[k])
                
            # axs[1].set_ylim([-500,12000])
            axs[1].set_xlabel('Area use coefficient [-]')
            axs[1].set_ylabel("Nominal capacity [GW]")
            
            if component == "hydrogen":
                axs[1].set_xlim(0,max(area_ranges[component]))
            if component == "data":
                axs[1].set_xlim(min(area_ranges[component]),max(area_ranges[component]))
            if component == "storage":
                axs[1].set_xlim(0,max(area_ranges[component]))
            
            axs[1].set_ylim([-750,15_750])
            axs[1].set_yticks(np.arange(0,16_000,3_000),[0,3,6,9,12,15])
            
            axs[1].yaxis.set_minor_locator(MultipleLocator(1000))
            axs[1].xaxis.set_minor_locator(MultipleLocator(0.1))
            
            axs_copy2 = axs[1].twinx()
            axs_copy2.set_ylim([-0.05,1.05])
            axs_copy2.set_ylabel('Normalized optimum [-]')
            
            area_optimum = area_sensitivity_cap[component]['Optimum'].copy()
            area_optimum[:] = [x / max(area_optimum) for x in area_optimum]
            axs_copy2.plot(x1, area_optimum, linestyle='-.', marker='.', markersize=4, color = 'k', label = 'Optimum', linewidth = 0.4)
            axs_copy2.yaxis.set_minor_locator(MultipleLocator(0.05))
            
            lines, labels = axs[i].get_legend_handles_labels()
            lines2, labels2 = axs_copy2.get_legend_handles_labels()
            axs[1].legend(lines + lines2, labels + labels2, loc='upper right', bbox_to_anchor=(1.01, 0.96), fontsize = 15)    
            
    # plt.tight_layout() 
    plt.savefig('../../images/sensitivity/' + str(year) + '_' + component + '_area_sensitivity.pdf', format = 'pdf', bbox_inches='tight')
plt.show()