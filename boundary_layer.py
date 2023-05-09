import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy import special
from scipy.optimize import curve_fit
import statistics
import re
import modules.island_plt as ip
ip.set_plot_options()
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

#%% Choose lot

lotn = 2
lot  = 'Lot ' + str(lotn)

#%% Get list with folders in working directory
folder = os.getcwd() + "/data/energinet/"

sub_folders = [name for name in os.listdir(folder) 
               if os.path.isdir(os.path.join(folder, name))
               and lot in name]

#%% Read data


# Initiate list of dataframes
wind_data_seperated = []

# Uncomment this line to run only a specified month
# sub_folders = ['2101_1_1 Lot 1 Month 1']

# Iterate through each folder
for i in sub_folders:  
    
# folder = sub_folders[0]

    # Get the file(s) in the folder which are files and have "WindStatus" in 
    # the name
    file_name = [file for file in os.listdir(folder + i) 
                      if os.path.isfile(os.path.join(folder + i, file)) 
                      and 'WindSpeed' in file]
    
    # If file_name is empty, skip this iteration
    if not bool(file_name):
        continue 
    # Combine folder and filename to file path
    file_path = folder + i + '/' +  file_name[0]
    
    # Create dataframe from file. Skip first 2 rows, set row 0 as index
    file_data  = pd.read_csv(file_path, sep = ';', skiprows = [0, 1],
                             index_col = 0)
    
    # Convert index to proper datetime
    file_data.index = pd.to_datetime(file_data.index)
    
    # Filter dataframe, to return only columns with "liPacketCount" in the name.
    # These columns are the ones with actual data.
    file_data = file_data.filter(regex = 'WindSpeed')
    
    file_data = file_data.iloc[:,-12:]
    
    # Append data from this file in this folder to the overall list of dataframes
    wind_data_seperated.append(file_data)

# Combine all dataframe in the list into one dataframe
wind_data = pd.concat(wind_data_seperated)
wind_data = wind_data.resample('H').mean()
wind_data.rename(columns={'WindSpeed004m m/s': int(4), 
                          'WindSpeed030m m/s': int(30),
                          'WindSpeed040m m/s': int(40),
                          'WindSpeed060m m/s': int(60),
                          'WindSpeed090m m/s': int(90),
                          'WindSpeed100m m/s': int(100),
                          'WindSpeed120m m/s': int(120),
                          'WindSpeed150m m/s': int(150),
                          'WindSpeed180m m/s': int(180),
                          'WindSpeed200m m/s': int(200),
                          'WindSpeed240m m/s': int(240),
                          'WindSpeed270m m/s': int(270),}, inplace=True)
#%% Mean data and boundary layer
mean_wind = wind_data.mean()

h = np.arange(4,270,1)
v = np.empty(len(h), dtype=object)
v[0] = mean_wind.values[0]
z0 = 0.0002

for i in range(len(h)-1):
    if i < len(h):
        v[i+1] = v[i]*((np.log(h[i+1]/z0))/(np.log(h[i]/z0)))
    else:
        break

fig, ax = plt.subplots(1,1, figsize = (10,5), dpi = 300)
ax.plot(mean_wind.values, mean_wind.index, label = 'DEA mean wind speed')
ax.plot(v, h, label = 'Model velocity profile')

ax.axhline(y=120, color='k', linestyle='--', linewidth = 1)
ax.axhline(y=150, color='k', linestyle='--', linewidth = 1)
ax.plot([v[120-4],v[150-4]], [120,150],color = 'r', linestyle="--", label = 'Linear stretch', linewidth = 1.5)

ax.set_xticks(np.arange(8.5,13,0.5))
ax.set_yticks(np.arange(0,350,50))

ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(10))

ax.set_xlim([8.5,12.5])
ax.set_ylim([0,300])

ax.set_title('Mean velocity from DEA & model velocity profile', pad = 15)
ax.set_xlabel('Wind speed [m/s]')
ax.set_ylabel('Height from sea level [m]')

ax.legend()