 # -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 20:21:54 2023

@author: lukas
"""
#%%
# ----- General Operation Resource Module (GORM) -----
# The GORM module contains useful helper functions for the main model.

#%% -------GENERAL FUNCTIONS -----------------------------------

# Play sound from folder
def its_britney_bitch(path):
    import os
    import random 

    files_in_path   = os.listdir(path) # Get files in directory
    file_to_play    = random.choice(files_in_path) # Choose random file to play
    
    # Play file
    os.startfile(path + '/' + file_to_play)
    
#%% ------- CALCULATION / OPERATION FUNCTIONS ---------------------------------

# Get annuity Annuity from discount rate (i) and lifetime (n)
def get_annuity(i, n):
    annuity = i/(1.-1./(1.+i)**n)
    return annuity

# Remove outliers from dataframe, and replace with n standard deviations.
def remove_outliers(df,columns,n_std):
    for col in columns:
        print('Removing outliers - Working on column: {}'.format(col))
        
        df[col][ df[col] >= df[col].mean() + (n_std * df[col].std())] = \
        df[col].mean() + (n_std * df[col].std())
        
    return df

def get_earth_distance(lat1,lat2,lon1,lon2):
    import numpy as np
    R = 6378.1  #Earths radius
    dlon = (lon2 - lon1) * np.pi/180
    dlat = (lat2 - lat1) * np.pi/180
    
    lat1 = lat1 * np.pi/180
    lat2 = lat2 * np.pi/180
    
    a = (np.sin(dlat/2))**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2))**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)) 
    d = R*c #Spherical distance between two points
    
    return d

#%% ------- PyPSA FUNCTIONS -----------------------------------

# Add bidirectional link with setup for losses
def add_bi_link(network, bus0, bus1, link_name, carrier, efficiency = 1,
               capital_cost = 0, marginal_cost = 0, 
               p_nom_extendable = True, p_nom_max = float('inf'),
               p_nom_min = 0,
               x = None, y = None, bus_shift = [0, 0]):
    # Function that adds a bidirectional link with efficiency and marginal cost
    # between two buses. This is done by adding additional "efficiency buses" 
    # (e0 and e1) for each bus, and running a bidirectional lossless link 
    # between them. The unidirectional links from e0 and e1 to their respective
    # buses then take care of adding correct marignal cost and efficiency.
    # capital cost is added on the bidirectional lossless link, while effieincy
    # and maringla cost are added on the links from efficiency buses to buses.
    
    # ---- Add efficiency buses at each end ----
    e0 = link_name + '_e0'
    e1 = link_name + '_e1'
    
    network.madd('Bus',
          names = [e0,  e1],
          x     = [network.buses.loc[network.buses.index == bus0]['x'] + bus_shift[0] , 
                   network.buses.loc[network.buses.index == bus1]['x'] + bus_shift[0]],
          y     = [network.buses.loc[network.buses.index == bus0]['y'] + bus_shift[1], 
                   network.buses.loc[network.buses.index == bus1]['y'] + bus_shift[1]],
          )
    
    # ---- main bidirectional Link ----
    # capital_cost and marginal_cost are applied here
    network.add('Link',
          link_name,
          bus0              = e0,
          bus1              = e1,
          p_min_pu          = -1,
          p_nom_extendable  = p_nom_extendable,
          capital_cost      = capital_cost,    #Capital cost is added here
          p_nom_max         = p_nom_max,
          p_nom_min         = p_nom_min,
          carrier           = carrier,
          )
    
    # ---- Links on bus 0 ----
    # link from buses to ebuses
    network.madd('Link',
          names         = [link_name + '_bus0_to_e0',  link_name + '_e0_to_bus0'],
          bus0          = [bus0,                       e0          ],
          bus1          = [e0,                         bus0        ],
          efficiency    = [1,                          efficiency  ],
          marginal_cost = [0,                          marginal_cost],
          p_nom_extendable  = [True, True],
          )
    
    # ---- Links on bus 0 ----
    # link from buses to ebuses
    network.madd('Link',
          names         = [link_name + '_bus1_to_e1',  link_name + '_e1_to_bus1'],
          bus0          = [bus1,                          e1          ],
          bus1          = [e1,                            bus1        ],
          efficiency    = [1,                             efficiency  ],
          marginal_cost = [0,                             marginal_cost],
          p_nom_extendable  = [True, True],
          )
    
#%% PLOTTING

def set_plot_options():
    import matplotlib.pyplot as plt
    import matplotlib
    color_bg      = "0.99"          #Choose background color
    color_gridaxe = "0.85"          #Choose grid and spine color
    rc = {"axes.edgecolor":color_gridaxe} 
    plt.style.use(('ggplot', rc))           #Set style with extra spines
    plt.rcParams['figure.dpi'] = 300        #Set resolution
    plt.rcParams['figure.figsize'] = [10, 5]
    matplotlib.rc('font', size=15)
    matplotlib.rc('axes', titlesize=20)
    matplotlib.rcParams['font.family'] = ['DejaVu Sans']     #Change font to Computer Modern Sans Serif
    plt.rcParams['axes.unicode_minus'] = False          #Re-enable minus signs on axes))
    plt.rcParams['axes.facecolor']= color_bg             #Set plot background color
    plt.rcParams.update({"axes.grid" : True, "grid.color": color_gridaxe}) #Set grid color
    plt.rcParams['axes.grid'] = True
    # plt.fontname = "Computer Modern Serif"