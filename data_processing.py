import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

#%% Choose lot

lotn = 4
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
    # wind_ninja = wind_ninja.drop(j)

#%% Linear interpolation
x1 = 120
y1 = wind_data['120m']
x2 = 150
y2 = wind_data['150m']
x  = 138
wind_data['138m'] = y1+(x-x1)*((y2-y1)/(x2-x1))

#%% Plot time-series data
h = '138m' # Chosen height

fig, ax = plt.subplots(2,3,figsize =(40, 10), dpi = 300)

ax[0,0].plot(test,   label = '2019 RenewableNinja')
ax[0,0].plot(wind_data[h], label = '2021 Energinet (LiDAR buoy)')

ax[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%b%d'))
ax[0,0].yaxis.set_minor_locator(MultipleLocator(1))

ax[0,0].set_ylim([0,35])
ax[0,0].set_xlim([wind_data[h].index[0], wind_data[h].index[-1]])

ax[0,0].set_xlabel('Time [hr]')
ax[0,0].set_ylabel('Wind speed [m/s]')

ax[0,0].legend(loc = 'upper right')
ax[0,0].grid()

ax[0,0].text(1.01, 1, 
                str('Renewable Ninja:\n'
                    + 'Std = ' + str(round(wind_ninja['wind_speed'].std(), 2)) +'m/s\n'
                    + 'Min = ' + str(round(wind_ninja['wind_speed'].min(), 2)) +'m/s\n'
                    + 'Mean = '+ str(round(wind_ninja['wind_speed'].mean(),2)) +'m/s\n'
                    + 'Max = ' + str(round(wind_ninja['wind_speed'].max(), 2)) +'m/s\n'),
                ha='left', va='top', 
                transform=ax[0,0].transAxes,
                )

ax[0,0].text(1.01, 0.75, 
                str('Energinet:\n'
                    + 'Std = ' + str(round(wind_data[h].std(), 2)) +'m/s\n'
                    + 'Min = ' + str(round(wind_data[h].min(), 2)) +'m/s\n'
                    + 'Mean = '+ str(round(wind_data[h].mean(),2)) +'m/s\n'
                    + 'Max = ' + str(round(wind_data[h].max(), 2)) +'m/s\n'),
                ha='left', va='top', 
                transform=ax[0,0].transAxes,
                )

ax[0,0].text(0.01, 0.97, 
                str('RÅ DATA\n'+str(lot)),
                ha='left', va='top', 
                transform=ax[0,0].transAxes,
                fontsize = 30,
                )

ax[0,0].set_title('Time series of wind speeds at '+ str(lot) +' at height ' + h,
              fontsize = 12)


ax[1,0].hist([wind_ninja['wind_speed'].values,wind_data[h].values], bins = np.arange(0,26,1), rwidth=0.9,label = ['2019 RenewableNinja',
                                                                                                             '2021 Energinet (LiDAR buoy)'])
ax[1,0].set_xticks(np.arange(0,26))
ax[1,0].legend()
ax[1,0].set_xlabel('Wind speed [m/s]')
ax[1,0].set_ylabel('Counts')

ax[1,0].text(0.01, 0.97, 
                str('RÅ DATA\n'+str(lot)),
                ha='left', va='top', 
                transform=ax[1,0].transAxes,
                fontsize = 30,
                )

ax[1,0].set_xlim([0,25])
ax[1,0].set_ylim([0,600])

ax[1,0].set_title('Histogram of wind speeds at ' + lot + ' and Renewable Ninja')

# Mean correction
wind_mean_corrected = wind_ninja['wind_speed']/wind_ninja['wind_speed'].mean() * wind_data[h].mean()

ax[0,1].plot(wind_mean_corrected, label = 'RenewableNinja (mean corrected)')
ax[0,1].plot(wind_data[h], label = '2021 Energinet (LiDAR buoy)')

ax[0,1].xaxis.set_major_formatter(mdates.DateFormatter('%b%d'))
ax[0,1].yaxis.set_minor_locator(MultipleLocator(1))

ax[0,1].set_ylim([0,35])
ax[0,1].set_xlim([wind_data[h].index[0], wind_data[h].index[-1]])

ax[0,1].set_xlabel('Time [hr]')
ax[0,1].set_ylabel('Wind speed [m/s]')

ax[0,1].legend(loc = 'upper right')
ax[0,1].grid()

ax[0,1].text(1.01, 1, 
                str('Renewable Ninja:\n'
                    + 'Std = ' + str(round(wind_mean_corrected.std(), 2)) +'m/s\n'
                    + 'Min = ' + str(round(wind_mean_corrected.min(), 2)) +'m/s\n'
                    + 'Mean = '+ str(round(wind_mean_corrected.mean(),2)) +'m/s\n'
                    + 'Max = ' + str(round(wind_mean_corrected.max(), 2)) +'m/s\n'),
                ha='left', va='top', 
                transform=ax[0,1].transAxes,
                )

ax[0,1].text(1.01, 0.75, 
                str('Energinet:\n'
                    + 'Std = ' + str(round(wind_data[h].std(), 2)) +'m/s\n'
                    + 'Min = ' + str(round(wind_data[h].min(), 2)) +'m/s\n'
                    + 'Mean = '+ str(round(wind_data[h].mean(),2)) +'m/s\n'
                    + 'Max = ' + str(round(wind_data[h].max(), 2)) +'m/s\n'),
                ha='left', va='top', 
                transform=ax[0,1].transAxes,
                )

ax[0,1].text(0.01, 0.97, 
                str('Mean\ncorrected\n'+lot),
                ha='left', va='top', 
                transform=ax[0,1].transAxes,
                fontsize = 30,
                )

ax[0,1].set_title('Time series of wind speeds at '+ str(lot) +' at height ' + h,
              fontsize = 12)

ax[1,1].hist([wind_mean_corrected,wind_data[h].values], 
         bins = np.arange(0,26,1), 
         rwidth = 0.9,
         label = ['RenewableNinja (mean corrected)',
                  '2021 Energinet (LiDAR buoy)',
                  ])

ax[1,1].set_xticks(np.arange(0,26))
ax[1,1].legend()
ax[1,1].set_xlabel('Wind speed [m/s]')
ax[1,1].set_ylabel('Counts')
ax[1,1].set_xlim([0,25])
ax[1,1].set_ylim([0,600])
ax[1,1].set_title('Histogram of wind speeds at ' + lot + ' and Renewable Ninja')

ax[1,1].text(0.01, 0.97, 
                str('Mean\ncorrected\n'+lot),
                ha='left', va='top', 
                transform=ax[1,1].transAxes,
                fontsize = 30,
                )

# Alt. std correction
# fig3, ax3 = plt.subplots(2,1,figsize =(10, 10), dpi = 300)

wind_mean_corrected = wind_ninja['wind_speed']/wind_ninja['wind_speed'].mean() * wind_data[h].mean()
wind_std_corrected = wind_ninja['wind_speed']/wind_ninja['wind_speed'].std() * wind_data[h].std()
wind_std_corrected = wind_std_corrected - wind_mean_corrected.std()
wind_std_corrected = wind_std_corrected[wind_std_corrected > 0]

ax[0,2].plot(wind_std_corrected, label = 'RenewableNinja (alt. std corrected)')
ax[0,2].plot(wind_data[h], label = '2021 Energinet (LiDAR buoy)')

ax[0,2].xaxis.set_major_formatter(mdates.DateFormatter('%b%d'))
ax[0,2].yaxis.set_minor_locator(MultipleLocator(1))

ax[0,2].set_ylim([0,35])
ax[0,2].set_xlim([wind_data[h].index[0], wind_data[h].index[-1]])

ax[0,2].set_xlabel('Time [hr]')
ax[0,2].set_ylabel('Wind speed [m/s]')

ax[0,2].legend(loc = 'upper right')
ax[0,2].grid()

ax[0,2].text(1.01, 1, 
                str('Renewable Ninja:\n'
                    + 'Std = ' + str(round(wind_std_corrected.std(), 2)) +'m/s\n'
                    + 'Min = ' + str(round(wind_std_corrected.min(), 2)) +'m/s\n'
                    + 'Mean = '+ str(round(wind_std_corrected.mean(),2)) +'m/s\n'
                    + 'Max = ' + str(round(wind_std_corrected.max(), 2)) +'m/s\n'),
                ha='left', va='top', 
                transform=ax[0,2].transAxes,
                )

ax[0,2].text(1.01, 0.75, 
                str('Energinet:\n'
                    + 'Std = ' + str(round(wind_data[h].std(), 2)) +'m/s\n'
                    + 'Min = ' + str(round(wind_data[h].min(), 2)) +'m/s\n'
                    + 'Mean = '+ str(round(wind_data[h].mean(),2)) +'m/s\n'
                    + 'Max = ' + str(round(wind_data[h].max(), 2)) +'m/s\n'),
                ha='left', va='top', 
                transform=ax[0,2].transAxes,
                )

ax[0,2].text(0.01, 0.97, 
                str('Alt. std\ncorrected\n'+lot),
                ha='left', va='top', 
                transform=ax[0,2].transAxes,
                fontsize = 30,
                )

ax[0,2].set_title('Time series of wind speeds at '+ str(lot) +' at height ' + h,
              fontsize = 12)

ax[1,2].hist([wind_std_corrected,wind_data[h].values], 
         bins = np.arange(0,26,1), 
         rwidth = 0.9,
         label = ['RenewableNinja (alt. std corrected)',
                  '2021 Energinet (LiDAR buoy)',
                  ])

ax[1,2].set_xticks(np.arange(0,26))
ax[1,2].legend()
ax[1,2].set_xlabel('Wind speed [m/s]')
ax[1,2].set_ylabel('Counts')
ax[1,2].set_ylim([0,600])
ax[1,2].set_xlim([0,25])
ax[1,2].set_title('Histogram of wind speeds at ' + lot + ' and Renewable Ninja')

ax[1,2].text(0.01, 0.97, 
                str('Alt. std\ncorrected\n'+lot),
                ha='left', va='top', 
                transform=ax[1,2].transAxes,
                fontsize = 30,
                )
