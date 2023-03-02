# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 18:58:40 2022

@author: lukas & anders
"""
#%% Import packages, set colors and set up plotting
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import style
import numpy as np
import cartopy.crs as ccrs
import pandas as pd

#%% Default colorscheme
def get_plot_colors():
    colors = {"IL":"black",
              "DK":"tab:red", 
              "NO":"forestgreen", 
              "DE":"gold", 
              "NE":"tab:orange", 
              "BE":"tab:brown", 
              "GB":"tab:blue"}
    
    return colors

#%% Default plotting options
def set_plot_options():
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

#%% Geographical plot
def plot_geomap(network, bounds = [-3, 12, 59, 50.5], size = (15,15)):
    #Plots geographical map with buses and links shown
    
    plt.rc("figure", figsize = size)   #Set plot resolution

    network.plot(
        color_geomap = True,            #Coloring on oceans
        boundaries = bounds,            #Boundaries of the plot as [x1,x2,y1,y2]
        projection=ccrs.EqualEarth()    #Choose cartopy.crs projection
        )

#%% Powerflow plot
def plot_powerflow(network, size = [16*2,9], colors = get_plot_colors()):
    #Plots power flow from island to country buses.
    fig_PF, axs_PF = plt.subplots(1, 2)  # Set up subplot with 4 plots
    #plt.figure(dpi=300)            # Set resolution
    axs_PF = axs_PF.ravel()
        
    for i in np.arange(0,2):
        network.links_t.p0.iloc[:,i].plot(
            ax = axs_PF[i],
            # figsize = size,
            title = network.links_t.p0.columns[i],
            # color = colors[list(colors)[i+1]],
            )
        
    fig_PF.tight_layout()


#%% Power consumption and power production plots
def plot_loads_generators(network, size = (12, 12), location = "upper left"):
    #Plots all power consumption and power generation over time.
    fig, axs = plt.subplots(2)  # Set up subplot
    #plt.figure(dpi=600)         # Set resolution
    
    #---- Subplot 1 ----
    try: 
        network.loads_t.p.plot(              #Plot loads 
            ax = axs[0],                     #Choose axis in subplot
            ylabel = "Consumed power, [MW]", #Set ylabel
            figsize = size,                  #Set scaling
            grid = True,
            )
    except TypeError:
        print('WARNING: island_plt.plot_loads_generators:  No load data to plot')
    
    #---- Subplot 2 ----
    try:
        network.generators_t.p.plot(      #Plot generated power
            ax      = axs[1],                    #Choose axis in subplot
            ylabel  = "Produced power [MW]", #Set ylabel
            figsize = size,               #Set scaling
            grid = True,
            )
    except TypeError:
        print('WARNING: island_plt.plot_loads_generators:  No generator data to plot')
        
    #---- Legend location loop ----
    for i in range(len(axs)): #Loop through subplots and change legend location
        axs[i].legend(loc=location) 

#%% Correlation Matrix

def plot_corr_matrix(data, title='', fsize = 20, size = (15,15),  vmin = 0 ):
    #Show correlation matrix heatmap with 
    corr = data.corr()
    
    mask = np.triu(np.ones_like(corr))
    
    plt.figure()
    
    cmap = sn.diverging_palette(10, 130, as_cmap=True)
    ax = sn.heatmap(corr,
               annot=True, mask = mask, square = True, cbar_kws={"shrink": .82},
               linewidth = 1, cmap = cmap, vmin = vmin)
    
    ax.set_title(title, fontsize = 20, fontweight = "bold",)
    
    plt.show()
    
#%% Price diff

def plot_price_diff(opt_prices, colors = get_plot_colors()):
    fig, axs = plt.subplots(6, sharey = True)  # Set up subplot
    country_price = opt_prices.iloc[:,1:]

    for i in range(len(country_price.columns)):
        #Calcultae difference between country and island price
        diff = country_price.iloc[:,i] - opt_prices.iloc[:,0]
        
        #plot, add plot color, add grid
        axs[i].plot(diff.values,
                    color = colors[list(colors)[i+1]], )
        
        #Set ylabel and title
        axs[i].set_ylabel("Euro/MWh")
        axs[i].set_title(str(country_price.columns[i]) + " price minus Island price")
        
        #Add fill between plot and horizontal line at 0.
        axs[i].fill_between(np.arange(len(diff)), diff,
                            color = colors[list(colors)[i+1]],
                            alpha = 0.2)
        
    #Set tight layout for better layout formatting    
    fig.tight_layout()
    
#%% Bus Prices
def plot_bus_prices(opt_prices, colors = get_plot_colors()):
    fig, axs = plt.subplots(len(opt_prices.columns), sharey = True)  # Set up subplot
    #.figure(dpi=600)         # Set resolution
    
    for i in range(len(opt_prices.columns)):
        axs[i].plot(opt_prices.iloc[:,i].values,
                    color = colors[list(colors)[i]]
                    )
        #axs[i].set_ylim([None, opt_prices.values.ravel().max()])
        
        axs[i].grid(True)
        axs[i].set_ylabel("Euro/MWh")
        axs[i].set_title(str(opt_prices.columns[i]) + " - Electricity price")
        axs[i].fill_between(np.arange(len(opt_prices)),opt_prices.iloc[:,i].values,
                            color = colors[list(colors)[i]],
                            alpha = 0.2)
        
    fig.tight_layout()
    
#%% Scatter matrix
def plot_scatter(df):
    mask = np.triu(np.ones_like((df.shape[1], df.shape[1])))
    
    pd.plotting.scatter_matrix(df,
            grid=False, color = "tab:blue", diagonal = "kde")
    
    
    
    
    
    
    
    
    
    