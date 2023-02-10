import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import island_plt as ip

ip.set_plot_options()

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


#%% Read Renewable Ninja

wind_ninja = pd.read_csv('ninja_wind.csv',
                          sep = ',',
                          index_col = 0,
                          usecols = [0,3],
                          comment = '#',
                          )
wind_ninja.index = pd.to_datetime(wind_data.index)

#%% Plot time-series data
fig, ax = plt.subplots(figsize =(10, 5), dpi = 300)
plt.plot(wind_ninja, label = '2019 RenewableNinja, (hub)height = 138m')
plt.plot(wind_data.iloc[:,5], 
          label = '2021 Energinet (LiDAR buoy), height = 120m')

# ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b%d'))
ax.yaxis.set_minor_locator(MultipleLocator(1))

# ax.set_xlim([wind_ninja.index[0], wind_ninja.index[-1]])
# ax.set_ylim(0,25)

ax.set_xlabel('Time [hr]')
ax.set_ylabel('Wind speed [m/s]')

plt.legend(loc = 'lower left')
plt.grid()

ax.text(1.01, 1, 
                str('Renewable Ninja:\n'
                    + 'Std = '+ str(round(wind_ninja['wind_speed'].std(),2)) +'m/s\n'
                    + 'Min = '+ str(round(wind_ninja['wind_speed'].min(),2)) +' m/s\n'
                    + 'Mean = '+ str(round(wind_ninja['wind_speed'].mean(),2)) +'m/s\n'
                    + 'Max = '+ str(round(wind_ninja['wind_speed'].max(),2)) +'m/s\n'),
                ha='left', va='top', 
                transform=ax.transAxes,
                )

ax.text(1.01, 0.8, 
                str('Statistics:\n'
                    + 'Std = '+ str(round(wind_data.iloc[:,5].std(),2)) +'m/s\n'
                    + 'Min = '+ str(round(wind_data.iloc[:,5].min(),2)) +' m/s\n'
                    + 'Mean = '+ str(round(wind_data.iloc[:,5].mean(),2)) +'m/s\n'
                    + 'Max = '+ str(round(wind_data.iloc[:,5].max(),2)) +'m/s\n'),
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

#%% Statistics

# def swap_rows(df, row1, row2):
#     df.iloc[row1], df.iloc[row2] =  df.iloc[row2].copy(), df.iloc[row1].copy()
#     return df

heights = [270, 240, 200, 180, 150, 120, 100, 90, 60, 30, 40, 4]
heights = list(reversed(heights))

std, min_, mean, max_ = [], [], [], []

for i in range(0, wind_data.shape[1]):
    std.append(round(wind_data.iloc[:,i].std(),2))
    min_.append(round(wind_data.iloc[:,i].min(),2))
    mean.append(round(wind_data.iloc[:,i].mean(),2))
    max_.append(round(wind_data.iloc[:,i].max(),2))

stats = pd.DataFrame({'height':heights,
                      'std':std,
                      'min':min_,
                      'mean':mean,
                      'max':max_})

# swap_rows(stats, -1, -2)

# stats = stats.iloc[::-1]


