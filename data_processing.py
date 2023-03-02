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
import modules.island_plt as ip
ip.set_plot_options()

#%% Choose lot

lotn = 1
lot  = 'Lot ' + str(lotn)
h = '138m' # Chosen height

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
wind_data.rename(columns={'WindSpeed004m m/s': '004m', 
                          'WindSpeed030m m/s': '030m',
                          'WindSpeed040m m/s': '040m',
                          'WindSpeed060m m/s': '060m',
                          'WindSpeed090m m/s': '090m',
                          'WindSpeed100m m/s': '100m',
                          'WindSpeed120m m/s': '120m',
                          'WindSpeed150m m/s': '150m',
                          'WindSpeed180m m/s': '180m',
                          'WindSpeed200m m/s': '200m',
                          'WindSpeed240m m/s': '240m',
                          'WindSpeed270m m/s': '270m',}, inplace=True)

#%% Read Renewable Ninja
folder = os.getcwd() + "/data/renewableninja/"

for file in os.listdir(folder): 
    if "lot"+str(lotn) in file:
        file_name = file
        break

wind_ninja = pd.read_csv(folder + file_name,
                          sep = ',',
                          index_col = 0,
                          usecols = [0,3],
                          comment = '#',
                          )
test = pd.DataFrame(0, index=np.arange(len(wind_data.index)), columns=['time','wind_speed'])
test.set_index("time", inplace = True)
test.index = pd.to_datetime(wind_data.index)

#%%
# test = pd.DataFrame.set_index()
for i in wind_data.index:
    for j in wind_ninja.index:
        if str(j)[5:-3] == str(i)[5:-12]:
            test["wind_speed"][i] = wind_ninja.loc[j].values
        else:
            continue
wind_ninja = test

#%% Linear interpolation
x1 = 120
y1 = wind_data['120m']
x2 = 150
y2 = wind_data['150m']
x  = 138
wind_data['138m'] = y1+(x-x1)*((y2-y1)/(x2-x1))

#%% Plot time-series and histogram data
fig, ax = plt.subplots(1,1,figsize = (10, 5), dpi = 300)

ax.plot(wind_data[h], label = '2021 Energinet (LiDAR buoy)')
ax.plot(wind_ninja,   label = '2019 RenewableNinja')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b%d'))
ax.yaxis.set_minor_locator(MultipleLocator(1))

ax.set_ylim([0,35])
ax.set_xlim([wind_data[h].index[0], wind_data[h].index[-1]])

ax.set_xlabel('Time [hr]')
ax.set_ylabel('Wind speed [m/s]')

ax.legend(loc = 'upper right', fontsize = 11)
ax.grid()

ax.text(1.01, 1, 
                str('Renewable Ninja:\n'
                    + 'Std = ' + str(round(wind_ninja['wind_speed'].std(), 1)) +'m/s\n'
                    + 'Min = ' + str(round(wind_ninja['wind_speed'].min(), 1)) +'m/s\n'
                    + 'Mean = '+ str(round(wind_ninja['wind_speed'].mean(),1)) +'m/s\n'
                    + 'Max = ' + str(round(wind_ninja['wind_speed'].max(), 1)) +'m/s\n'),
                ha='left', va='top', fontsize=11, 
                transform=ax.transAxes,
                )

ax.text(1.01, 0.65, 
                str('Energinet:\n'
                    + 'Std = ' + str(round(wind_data[h].std(), 1)) +'m/s\n'
                    + 'Min = ' + str(round(wind_data[h].min(), 1)) +'m/s\n'
                    + 'Mean = '+ str(round(wind_data[h].mean(),1)) +'m/s\n'
                    + 'Max = ' + str(round(wind_data[h].max(), 1)) +'m/s\n'),
                ha='left', va='top', fontsize = 11, 
                transform=ax.transAxes,
                )

ax.set_title('Time series of wind speeds at '+ str(lot) +' at height ' + h)

fig, ax = plt.subplots(1,1,figsize = (10, 5), dpi = 300)

ax.hist([wind_data[h].values,wind_ninja['wind_speed'].values], bins = np.arange(0,27,1), rwidth=0.9,label = ['2021 Energinet (LiDAR buoy)','2019 RenewableNinja'])
ax.set_xticks(np.arange(0,27))
ax.legend(fontsize = 11)
ax.set_xlabel('Wind speed [m/s]')
ax.set_ylabel('Counts')

ax.set_xlim([0,26])
ax.set_ylim([0,600])

ax.set_title('Histogram of wind speeds at ' + lot + ' and Renewable Ninja')
