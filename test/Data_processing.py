import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

#%% Choose lot

lot = 1
lot = 'Lot ' + str(lot)

#%% Get list with folders in working directory
folder = os.getcwd()
sub_folders = [name for name in os.listdir(folder) 
               if os.path.isdir(os.path.join(folder, name))
               and lot in name]

#%% Read data

# Initiate list of dataframes
wind_data_seperated = []

# Uncomment this line to run only a specified month
# sub_folders = ['2101_1_1 Lot 1 Month 1']

# Iterate through each folder
for folder in sub_folders:  
    
# folder = sub_folders[0]

    # Get the file(s) in the folder which are files and have "WindStatus" in 
    # the name
    file_name = [file for file in os.listdir(folder) 
                      if os.path.isfile(os.path.join(folder, file)) 
                      and 'WindSpeed' in file]
    
    # If file_name is empty, skip this iteration
    if not bool(file_name):
        continue 
    # Combine folder and filename to file path
    file_path = folder + '/' +  file_name[0]
    
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

wind_ninja = pd.read_csv('ninja_wind.csv',
                          sep = ',',
                          index_col = 0,
                          usecols = [0,3],
                          comment = '#',
                          )
wind_ninja.index = pd.to_datetime(wind_data.index)

#%% Plot time-series data
h = '120m' # Chosen height

fig, ax = plt.subplots(figsize =(10, 5), dpi = 300)

plt.plot(wind_ninja,   label = '2019 RenewableNinja, (hub)height = 138m')
plt.plot(wind_data[h], label = '2021 Energinet (LiDAR buoy), height = 120m')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b%d'))
ax.yaxis.set_minor_locator(MultipleLocator(1))

ax.set_xlabel('Time [hr]')
ax.set_ylabel('Wind speed [m/s]')

plt.legend(loc = 'upper right')
plt.grid()

ax.text(1.01, 1, 
                str('Renewable Ninja:\n'
                    + 'Std = ' + str(round(wind_ninja['wind_speed'].std(), 2)) +'m/s\n'
                    + 'Min = ' + str(round(wind_ninja['wind_speed'].min(), 2)) +'m/s\n'
                    + 'Mean = '+ str(round(wind_ninja['wind_speed'].mean(),2)) +'m/s\n'
                    + 'Max = ' + str(round(wind_ninja['wind_speed'].max(), 2)) +'m/s\n'),
                ha='left', va='top', 
                transform=ax.transAxes,
                )

ax.text(1.01, 0.75, 
                str('Energinet:\n'
                    + 'Std = ' + str(round(wind_data[h].std(), 2)) +'m/s\n'
                    + 'Min = ' + str(round(wind_data[h].min(), 2)) +'m/s\n'
                    + 'Mean = '+ str(round(wind_data[h].mean(),2)) +'m/s\n'
                    + 'Max = ' + str(round(wind_data[h].max(), 2)) +'m/s\n'),
                ha='left', va='top', 
                transform=ax.transAxes,
                )

ax.set_title('Time series of wind speeds at 120m \n  Lot 1, December 2021, WindStatus',
              fontsize = 12)

#%% Plot histogram
fig, ax = plt.subplots(figsize =(14, 7), dpi = 300)
ax.hist([wind_ninja['wind_speed'].values,wind_data.iloc[:,5].values], bins = np.arange(0,26,1), rwidth=0.9,label = ['2019 RenewableNinja, (hub)height = 138m',
                                                                                                                    '2021 Energinet (LiDAR buoy), height = 120m'])

# ax.set_xlim([-0.5,25.5])
plt.xticks(np.arange(0,26))
plt.legend()
# ax.set_ylim([0,1600])
ax.set_xlabel('Wind speed [m/s]')
ax.set_ylabel('Counts')

ax.set_title('Histogram of wind speeds at latitude = 56.6280, longitude = 6.3007 (Energy Island)')

#%% Correction

wind_corrected = wind_ninja['wind_speed']/wind_ninja['wind_speed'].mean() * wind_data[h].mean()
# wind_corrected = wind_ninja['wind_speed']/wind_ninja['wind_speed'].std() * wind_data[h].std()
# wind_corrected = wind_corrected - wind_corrected.std()
fig, ax = plt.subplots(figsize =(10, 5), dpi = 300)

plt.plot(wind_data[h],   label = 'Energinet')
plt.plot(wind_corrected, label = 'Corrected')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b%d'))
ax.yaxis.set_minor_locator(MultipleLocator(1))

ax.set_xlabel('Time [hr]')
ax.set_ylabel('Wind speed [m/s]')

plt.legend(loc = 'upper right')
plt.grid()


print(wind_data[h].describe())
print(wind_corrected.describe())

fig, ax = plt.subplots(figsize =(14, 7), dpi = 300)
ax.hist([wind_data[h].values,wind_corrected.values], bins = np.arange(0,26,1), rwidth=0.9,label = ['Energinet','Corrected'])
plt.legend()