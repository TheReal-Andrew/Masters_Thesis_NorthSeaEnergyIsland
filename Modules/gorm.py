# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 20:21:54 2023

@author: lukas
"""
#%%
# General cOde Resource Module (GORM) - module containing useful functions

#%% General functions

# Play sound from folder
def its_britney_bitch(path):
    import os
    import random 
    path=path
    files=os.listdir(path)
    d=random.choice(files)
    os.startfile(path + '/' + d)

#%% Calculations and Operations

# Get annuity Annuity from discount rate and lifetime
def get_annuity(i, n):
    annuity = i/(1.-1./(1.+i)**n)
    return annuity

# Remove outliers from dataframe, and replace with n standard deviations.
def remove_outliers(df,columns,n_std):
    for col in columns:
        print('Working on column: {}'.format(col))
        
        df[col][ df[col] >= df[col].mean() + (n_std * df[col].std())] = \
        df[col].mean() + (n_std * df[col].std())
        
    return df

#%% PyPSA Functions

# Add bidirectional link with setup for losses
def add_bilink(n, bus0, bus1, link_name, efficiency = 1,
               capital_cost = 0, marginal_cost = 0, p_nom_extendable = True):
    
    # ---- Add efficiency buses at each end ----
    n.madd('Bus',
          [link_name + '_ebus0',
           link_name + '_ebus1',
           ],
          )
    
    # ---- main bidirectional Link ----
    # capital_cost and marginal_cost are applied here
    n.add('Link',
          link_name,
          bus0              = link_name + '_ebus0',
          bus1              = link_name + '_ebus1',
          p_min_pu          = -1,
          p_nom_extendable  = p_nom_extendable,
          capital_cost      = capital_cost,
          marginal_cost     = marginal_cost,
          )
    
    # ---- links from buses to ebuses ----
    # Efficiency is applied here
    n.madd('Link',
          [link_name + '_ebus0_elink0',
           link_name + '_ebus1_elink0'
           ],
          bus0              = [bus0, bus1],
          bus1              = [link_name + '_ebus0',
                               link_name + '_ebus1'
                               ],
          p_nom_extendable  = True,
          )
    
    #links from ebuses to buses
    n.madd('Link',
          [link_name + '_ebus0_elink1',
           link_name + '_ebus1_elink1'
           ],
          bus0              = [link_name + '_ebus0',
                               link_name + '_ebus1'
                               ],
          bus1              = [bus0, bus1],
          efficiency        = efficiency,
          p_nom_extendable  = True,
          )