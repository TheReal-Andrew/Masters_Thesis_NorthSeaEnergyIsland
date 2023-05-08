"""
Created on Tue Sep  6 10:44:50 2022

@author: Lukas & Anders
"""

import sys
sys.path.append("../../")
import Modules.island_lib as il #Library with plotting functions.
import Modules.island_plt as ip #Library with plotting functions.
# importlib.reload(il)
import pypsa
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import os
import datetime
ip.set_plot_options()

n_points = 8760

# Load power-price and consumer-load data
cprice, cload = il.get_load_and_price(2030)

cprice = il.remove_outliers(cprice,['DK','BE'],1)
cload = il.remove_outliers(cload,['DK','BE'],1)

# Load wind capacity factor (CF)
cf_wind_df = pd.read_csv(r'../../Data/Wind/wind_formatted.csv', sep = ",")

#Load technology data
tech_cost = pd.read_csv('https://github.com/PyPSA/technology-data/blob/master/inputs/manual_input.csv?raw=true')
tech_inv  = tech_cost.loc[tech_cost['parameter'].str.startswith('investment') & tech_cost['technology'].str.startswith('HVDC submarine')]
tech_life = tech_cost.loc[tech_cost['parameter'].str.startswith('lifetime') & tech_cost['technology'].str.startswith('HVDC submarine')]
tech_FOM  = tech_cost.loc[tech_cost['parameter'].str.startswith('FOM') & tech_cost['technology'].str.startswith('HVDC submarine')]

#%% Set up network with chosen timesteps
network = pypsa.Network()                   
t = pd.date_range('2030-01-01 00:00', '2030-12-31 23:00', freq = 'H')
network.set_snapshots(t)  #Set snapshots of network to the timesteps

#Create dataframe with info on buses. Names, x (Longitude) and y (Latitude) 
bus_df = pd.DataFrame(
    np.array([                          #Create numpy array with bus info
    ["Island",  "EI",   6.68, 56.52],   #Energy Island
    ["Denmark", "DK",   8.12, 56.37],   #Assumed Thorsminde
    ["Belgium", "BE",   3.20, 51.32]]), #Assumed Zeebrugge 
    columns = ["Country",               #Give columns titles
               "Abbreviation", 
               "X", 
               "Y"])

#%% Add buses -----------------------------------------------------
#Adding busses from bus_info to network
for i in range(bus_df.shape[0]):    #i becomes integers
    network.add(                    #Add component
        "Bus",                      #Component type
        bus_df.Country[i],          #Component name
        x = bus_df.X[i],            #Longitude (for plotting)
        y = bus_df.Y[i])            #Latitude (for plotting)

#%% Add links -----------------------------------------------------

#List of link destionations from buses
link_destinations = network.buses.index.values

DR = 1.2
j = 0
for i in link_destinations[1:]:         #i becomes each string in the array
    j = j + 1
    network.add(                        #Add component
        "Link",                         #Component type
        "Island_to_" + i,               #Component name
        bus0             = "Island",    #Start Bus
        bus1             = i,           #End Bus
        carrier          = "DC" + str(j),        #Define carrier type
        p_min_pu         = 0,          #Make links bi-directional
        p_nom            = 0,           #Power capacity of link
        p_nom_extendable = True,        #Extendable links
        capital_cost     = il.get_annuity(0.07, float(tech_life.value))    #Annuity factor
                            * float(tech_inv.value)                        #Investment cost [EUR/MW/km] 
                            * DR
                            * il.earth_distance(float(bus_df.X.loc[0]),  
                                                float(bus_df.X.loc[j]), 
                                                float(bus_df.Y.loc[0]), 
                                                float(bus_df.Y.loc[j])))

#%% Add Generators -------------------------------------------------

#Add wind turbine
network.add(
    "Generator",                        #Component type
    "Wind",                             #Component name
    bus           = "Island",           #Bus on which component is
    p_nom         = 3000,               #Nominal power [MW]
    p_max_pu      = cf_wind_df['electricity'].values, #Time-series of power coefficients
    carrier       = "Wind",             #Carrier type (AC,DC,Wind,Solar,etc.)
    marginal_cost = 20)                 #Cost per MW from this source 
    
#%% Add Generators -------------------------------------------------

#Add generators to each country bus with varying marginal costs
for i in range(1, bus_df.shape[0]):
    network.add(
        "Generator",
        "Gen_" + bus_df.Country[i],
        bus              = bus_df.Country[i],
        p_nom            = 0, 
        p_nom_extendable = True,         #Make sure country can always deliver to price
        capital_cost     = 0,            #Same as above
        marginal_cost    = cprice[bus_df.Abbreviation[i]].values
        )

#%% Add Loads ------------------------------------------------------

# Add loads to each country bus
for i in range(1, bus_df.shape[0]): #i becomes integers
    network.add(
        "Load",
        "Load_" + bus_df.Country[i],
        bus     = bus_df.Country[i],
        p_set   = cload[bus_df.Abbreviation[i]].values)       
   
#%% Plotting -------------------------------------------------------

# Geographical map
plt.rc("figure", figsize=(15, 15))    #Set plot resolution

network.plot(
    color_geomap = True,              #Coloring on oceans
    boundaries = [-3, 12, 59, 50.5],  #Boundaries of the plot as [x1,x2,y1,y2]
    projection=ccrs.EqualEarth()      #Choose cartopy.crs projection
    )

#%% Solver
network.lopf(pyomo = False,
             solver_name = 'gurobi')

#%% Save network

filename = '/case2_bidirect.nc'
export_path = os.getcwd() + filename
network.export_to_netcdf(export_path)

#%%
mean_DK7 = network.links_t.p0.iloc[:,0].resample('W').mean()
mean_BE7 = network.links_t.p0.iloc[:,1].resample('W').mean()

fig_PF, ax_PF = plt.subplots(2, 1, figsize=(16,9), dpi=300)

plt.sca(ax_PF[0])
plt.xticks(fontsize=15, rotation = 45) 
plt.yticks(fontsize=15)
ax_PF[0].plot(network.links_t.p0.iloc[:,0],
              label='Hourly',
              color = ip.get_plot_colors()[list(ip.get_plot_colors())[1]])
ax_PF[0].plot(mean_DK7, color='k', 
              linewidth = 3,
              label = 'Weekly')
ax_PF[0].set_xlabel('Time [hr]', fontsize = 15)
ax_PF[0].set_ylabel('Powerflow [MW]', fontsize = 15)
ax_PF[0].set_xlim([datetime.date(2030, 1, 1), datetime.date(2030,12,31)])
ax_PF[0].set_title('Direct powerflow from Energy Island to Denmark',
                    fontsize = 25)
ax_PF[0].set_ylim(-100,3000)
ax_PF[0].legend(loc="upper right",
                fontsize = 15)

ax_PF[0].text(1.01, 1, 
               str(round(network.links_t.p0.iloc[:,0].describe(),1).reset_index().to_string(header=None, index=None)),
               ha='left', va='top', 
               transform=ax_PF[0].transAxes,
               fontsize = 14)

ax_PF[0].grid()
plt.tight_layout()

plt.sca(ax_PF[1])
plt.xticks(fontsize=15, rotation = 45)
plt.yticks(fontsize=15)
ax_PF[1].plot(network.links_t.p0.iloc[:,1],
              color = ip.get_plot_colors()[list(ip.get_plot_colors())[5]],
              label = 'Hourly')
ax_PF[1].plot(mean_BE7, color='k', 
              linewidth = 3,
              label = 'Weekly')
ax_PF[1].set_xlabel('Time [hr]', fontsize = 15)
ax_PF[1].set_ylabel('Powerflow [MW]', fontsize = 15)
ax_PF[1].set_xlim([datetime.date(2030, 1, 1), datetime.date(2030,12,31)])
ax_PF[1].set_title('Direct powerflow from Energy Island to Belgium',
                    fontsize = 25)
ax_PF[1].set_ylim(-100,3000)
ax_PF[1].legend(loc="lower right",
                fontsize = 15)

ax_PF[1].text(1.01, 1, 
               str(round(network.links_t.p0.iloc[:,1].describe(),1).reset_index().to_string(header=None, index=None)), 
               ha='left', va='top', 
               transform=ax_PF[1].transAxes,
               fontsize = 14)

ax_PF[1].grid()
plt.tight_layout()


#%%
cprice.index = pd.date_range('2030-01-01 00:00', '2030-12-31 23:00', freq = 'H')
mean_priceDK7 = cprice['DK'].resample('D').mean()
mean_priceBE7 = cprice['BE'].resample('D').mean()

fig_PF1, ax_PF1 = plt.subplots(2, 1, figsize=(16,9), dpi=300)
ip.set_plot_options()

plt.sca(ax_PF[0])
plt.xticks(fontsize=15) 
plt.yticks(fontsize=15)
ax_PF1[0].plot(cprice['DK'],
                color = ip.get_plot_colors()[list(ip.get_plot_colors())[1]], label = 'Hourly')
ax_PF1[0].plot(mean_priceDK7, color = 'k', linewidth = 3, label = 'Daily')
ax_PF1[0].set_xlabel('Time [hr]', fontsize = 15)
ax_PF1[0].set_ylabel("Energy price [€/MWh]", fontsize = 15)
ax_PF1[0].set_xlim('2030-01-01 00:00', '2030-12-31 23:00')
ax_PF1[0].set_ylim(0,220)
ax_PF1[0].set_title('Danish energy price for 2030',
                    fontsize = 25)
ax_PF1[0].grid()
ax_PF1[0].text(1,1, 
               str(round(cprice['DK'].describe(),1).reset_index().to_string(header=None, index=None)),  
               ha='left', va='top', 
               transform=ax_PF1[0].transAxes,
               fontsize = 14)
ax_PF1[0].legend(loc="upper right",
                fontsize = 15)


plt.sca(ax_PF1[1])
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
ax_PF1[1].plot(cprice['BE'],
                color = ip.get_plot_colors()[list(ip.get_plot_colors())[5]], label = 'Hourly')
ax_PF1[1].plot(mean_priceBE7, color = 'k', linewidth = 3, label = 'Daily')
ax_PF1[1].set_xlabel('Time [hr]', fontsize = 15)
ax_PF1[1].set_ylabel("Energy price [€/MWh]", fontsize = 15)
ax_PF1[1].set_xlim('2030-01-01 00:00', '2030-12-31 23:00')
ax_PF1[1].set_ylim(0,220)
ax_PF1[1].set_title('Belgian energy price for 2030',
                    fontsize = 25)
ax_PF1[1].grid()
ax_PF1[1].text(1,1, 
               str(round(cprice['BE'].describe(),1).reset_index().to_string(header=None, index=None)), 
               ha='left', va='top', 
               transform=ax_PF1[1].transAxes,
               fontsize = 14)
ax_PF1[1].legend(loc="upper right",
                fontsize = 15)

plt.tight_layout()

#%% --------------------------------------------------------------------------
# cload.index = cload.index.astype("datetime64[ns]")

mean_demandDK7 = cload['DK'].resample('W').mean()
mean_demandBE7 = cload['BE'].resample('W').mean()

fig_PF2, ax_PF2 = plt.subplots(2, 1, figsize=(16,9), dpi=300)
ip.set_plot_options()

plt.sca(ax_PF2[0])
plt.xticks(fontsize=15) 
plt.yticks(fontsize=15)
ax_PF2[0].plot(cload['DK'],
                color = ip.get_plot_colors()[list(ip.get_plot_colors())[1]],
                label = 'Hourly')
ax_PF2[0].plot(mean_demandDK7,
               color = 'k',
               linewidth = 3,
               label = 'Weekly')
ax_PF2[0].set_ylabel('Energy demand [MWh]', fontsize = 15)
ax_PF2[0].set_xlim([datetime.date(2015, 1, 1), datetime.date(2015, 12, 31)])
ax_PF2[0].set_ylim(1900,10500)
ax_PF2[0].set_title('Danish energy demand for 2030',
                    fontsize = 25)
ax_PF2[0].grid()
ax_PF2[0].text(1.01, 1, 
               str(round(cload['DK'].describe(),1).reset_index().to_string(header=None, index=None)), 
               weight = 'heavy',
                ha='left', va='top', 
                transform=ax_PF2[0].transAxes,
               fontsize = 14)
ax_PF2[0].legend(loc="upper right",
                fontsize = 15)

plt.sca(ax_PF2[1])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ax_PF2[1].plot(cload['BE'],
                color = ip.get_plot_colors()[list(ip.get_plot_colors())[5]],
                label = 'Hourly')
ax_PF2[1].plot(mean_demandBE7,
               color = 'k',
               linewidth = 3,
               label = 'Weekly')
ax_PF2[1].set_xlabel('Time [hr]', fontsize = 15)
ax_PF2[1].set_ylabel('Energy demand [MWh]', fontsize = 15)
ax_PF2[1].set_xlim([datetime.date(2015, 1, 1), datetime.date(2015, 12, 31)])
ax_PF2[1].set_ylim(1900,10500)
ax_PF2[1].set_title('Belgian energy demand for 2030',
                    fontsize = 25)
ax_PF2[1].grid()
ax_PF2[1].text(1.01, 1, 
               str(round(cload['BE'].describe(),1).reset_index().to_string(header=None, index=None)), 
               weight = 'heavy',
                ha='left', va='top', 
                transform=ax_PF2[1].transAxes,
               fontsize = 14,)
ax_PF2[1].legend(loc="lower right",
                fontsize = 15)

plt.tight_layout()

#%% --------------------------------------------------------------------------
cf_wind_df['electricity'].index = pd.date_range('2030-01-01 00:00', '2030-12-31 23:00', freq = 'H')
mean_wind7 = cf_wind_df['electricity'].resample('W').mean()

plt.figure(figsize=(16,4.5), dpi=300)
ip.set_plot_options()
# plt.sca(ax_PF3)
plt.xticks(fontsize=15) 
plt.yticks(fontsize=15)
plt.plot(cf_wind_df['electricity'], color='blue', label = 'Hourly')
plt.plot(mean_wind7, color='k', linewidth = 3, label = 'Weekly')
# plt.plot(mean_wind7, color='k')
plt.xlabel('Time [hr]', fontsize = 15)
plt.ylabel('Wind capacity factor [-]', fontsize = 15)
plt.xlim('2030-01-01 00:00', '2030-12-31 23:00')
# ax_PF3.set_ylim(1900,10500)
plt.title('Wind capacity factor for 2030',
                    fontsize = 25)
plt.grid()
plt.text('2030-12-31 23:00',0.5, 
                str(round(cf_wind_df['electricity'].describe(),3).reset_index().to_string(header=None, index=None)),
                fontsize = 14)
plt.legend(loc = "lower right", fontsize = 15)


#%%
il.its_britney_bitch(r"C:\Users\aalin\Documents\GitHub\NorthSeaEnergyIsland\Data\Sounds")