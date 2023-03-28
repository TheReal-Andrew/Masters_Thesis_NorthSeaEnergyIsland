import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import modules.island_plt as ip
ip.set_plot_options()
from scipy import integrate
from scipy import interpolate
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

#%% Choose lot

lotn = 2
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

full_wind_ninja = pd.read_csv(folder + file_name,
                          sep = ',',
                          index_col = 0,
                          usecols = [0,3],
                          comment = '#',
                          )
cut_wind_ninja = pd.DataFrame(0, index=np.arange(len(wind_data.index)), columns=['time','wind_speed'])
cut_wind_ninja.set_index("time", inplace = True)
cut_wind_ninja.index = pd.to_datetime(wind_data.index)

#%% Split Renewable Ninja into DEA timeframe

for i in wind_data.index:
    for j in full_wind_ninja.index:
        if str(j)[5:-3] == str(i)[5:-12]:
            cut_wind_ninja["wind_speed"][i] = full_wind_ninja.loc[j].values
        else:
            continue

#%% Linear interpolation

#Fill Nan with mean value
wind_data['120m'] = wind_data['120m'].replace(np.nan,wind_data['120m'].mean())
wind_data['150m'] = wind_data['150m'].replace(np.nan,wind_data['150m'].mean())

x1 = 120
y1 = wind_data['120m']
x2 = 150
y2 = wind_data['150m']
x  = 138
wind_data['138m'] = y1+(x-x1)*((y2-y1)/(x2-x1))

#%% Plot time-series
fig, ax = plt.subplots(1,1,figsize = (10, 5), dpi = 300)

ax.plot(wind_data[h], label = '2021 Energinet (LiDAR buoy)')
ax.plot(cut_wind_ninja,   label = '2019 RenewableNinja')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b%d'))
ax.yaxis.set_minor_locator(MultipleLocator(1))

ax.set_ylim([0,35])
ax.set_xlim([wind_data[h].index[0], wind_data[h].index[-1]])

ax.set_xlabel('Time [hr]')
ax.set_ylabel('Wind speed [m/s]')
ax.set_title('Time series of wind speeds at ' + str(lot) + ' at height ' + h)

ax.legend(loc = 'upper right', fontsize = 11)
ax.grid(linewidth=1)

ax.text(1.01, 1, 
                str('Renewables.Ninja:\n'
                    + 'Std = ' + str(round(cut_wind_ninja['wind_speed'].std(), 1)) +'m/s\n'
                    + 'Min = ' + str(round(cut_wind_ninja['wind_speed'].min(), 1)) +'m/s\n'
                    + 'Mean = '+ str(round(cut_wind_ninja['wind_speed'].mean(),1)) +'m/s\n'
                    + 'Max = ' + str(round(cut_wind_ninja['wind_speed'].max(), 1)) +'m/s\n'),
                ha='left', va='top', fontsize=11, 
                transform=ax.transAxes,
                )

ax.text(1.01, 0.70, 
                str('Energinet:\n'
                    + 'Std = ' + str(round(wind_data[h].std(), 1)) +'m/s\n'
                    + 'Min = ' + str(round(wind_data[h].min(), 1)) +'m/s\n'
                    + 'Mean = '+ str(round(wind_data[h].mean(),1)) +'m/s\n'
                    + 'Max = ' + str(round(wind_data[h].max(), 1)) +'m/s\n'),
                ha='left', va='top', fontsize = 11, 
                transform=ax.transAxes,
                )

#%% Mean correction
full_wind_mean = full_wind_ninja/full_wind_ninja.mean() * wind_data[h].mean()

cut_wind_ninja_mean = pd.DataFrame(0, index=np.arange(len(wind_data.index)), columns=['time','wind_speed'])
cut_wind_ninja_mean.set_index("time", inplace = True)
cut_wind_ninja_mean.index = pd.to_datetime(wind_data.index)

for i in wind_data.index:
    for j in full_wind_mean.index:
        if str(j)[5:-3] == str(i)[5:-12]:
            cut_wind_ninja_mean["wind_speed"][i] = full_wind_mean.loc[j].values
        else:
            continue

#%% Mean correction histogram
fig, ax = plt.subplots(1,1,figsize = (10, 5), dpi = 300)

ax.hist([wind_data[h].values,
        cut_wind_ninja['wind_speed'].values, 
        cut_wind_ninja_mean['wind_speed'].values],
        bins = np.arange(0,27,1), 
        rwidth=0.9,
        label = ['Energinet',
                 'Renewables.Ninja',
                 'Mean corrected'])

ax.set_xticks(np.arange(0,27))
ax.legend(fontsize = 11)
ax.set_xlabel('Wind speed [m/s]')
ax.set_ylabel('Counts')

ax.set_xlim([0,26])
ax.set_ylim([0,600])

ax.set_title('Histogram of wind speeds at ' + lot)

ax.text(1.01, 1, 
                str('Mean corrected:\n'
                    + 'Std = ' + str(round(cut_wind_ninja_mean['wind_speed'].std(), 1)) +'m/s\n'
                    + 'Min = ' + str(round(cut_wind_ninja_mean['wind_speed'].min(), 1)) +'m/s\n'
                    + 'Mean = '+ str(round(cut_wind_ninja_mean['wind_speed'].mean(),1)) +'m/s\n'
                    + 'Max = ' + str(round(cut_wind_ninja_mean['wind_speed'].max(), 1)) +'m/s\n'),
                ha='left', va='top', fontsize = 11, 
                transform=ax.transAxes,
                )

#%% Mean correction QQ plot
fig, ax = plt.subplots(1,1,figsize = (5, 5), dpi = 300)

x = cut_wind_ninja.sort_values(by='wind_speed')
y = wind_data[h].sort_values()

x2 = cut_wind_ninja_mean.sort_values(by='wind_speed')
y2 = wind_data[h].sort_values()

ax.plot(x,y, label = 'Raw data')
ax.plot(x2,y2, label = 'Mean corrected')

ax.plot([0,35],[0,35],color = 'k', linestyle="--", linewidth = 1.5)

ax.set_xlim([0,35])
ax.set_ylim([0,35])

ax.set_xticks(np.arange(0,40,5))
ax.set_yticks(np.arange(0,40,5))

ax.set_title('DEA vs Renewables.Ninja')
ax.set_xlabel('Renewables.Ninja quantiles [m/s]')
ax.set_ylabel('DEA quantiles [m/s]')

ax.legend()

#%% Wind speeds to CF
def gaussian(v,v0):
    sigma = 0.12*v0 + 0.896 # IEC standard class A
    return np.exp(-0.5*((v-v0)/sigma)**2)/sigma/np.sqrt(2*np.pi)

class Wind_Turbine:
    def __init__(self, rated_power, cutin_speed, rated_speed, cutout_speed, power, velocity):
        self.rated_power  = rated_power
        self.cutin_speed  = cutin_speed
        self.rated_speed  = rated_speed
        self.cutout_speed = cutout_speed
        self.partial_pc   = interpolate.interp1d(velocity, power, 'cubic', fill_value = 'extrapolate')
    
    def single_pc(self, v):
        v = abs(v)
        return (0.*(v<=self.cutin_speed) + self.partial_pc(v)*(v>self.cutin_speed)*(v<=self.rated_speed)
                + self.rated_power*(v>self.rated_speed)*(v<=self.cutout_speed) + 0.*(v>self.cutout_speed))
    
    def multi_pc(self, v0):
        return integrate.quad_vec(lambda v: self.single_pc(v)*gaussian(v,v0), -np.inf, np.inf)[0]
    
# Vestas V164/8000
P = [0,40,100,370,650,895,1150,1500,1850,2375,2900,3525,4150,4875,5600,6350,7100,7580,7800,7920,8000] # kW
v = np.arange(3,13.5,0.5) # m/s
vestas = Wind_Turbine(8000,3,13,25,P,v)

# Renewables.ninja mean corrected
v = full_wind_mean['wind_speed'].values
P = vestas.multi_pc(v)
cf_wind = P/vestas.rated_power
    
pd.Series(cf_wind).to_csv('data/renewableninja/wind_cf_'+ lot +'.csv')

#%% Plot power curves
x = np.linspace(0,36,100)
fig, ax = plt.subplots(figsize = (10,5), dpi = 300)

ax.plot(x, vestas.single_pc(x)/10**3, label = 'Single turbine')
ax.plot(x, vestas.multi_pc(x)/10**3, label = 'Multiple turbines')

ax.set_xlabel('Wind speed [m/s]')
ax.set_ylabel('Power output [MW]')

ax.set_xlim([0,35])
ax.set_ylim([0,9])

plt.xticks(np.arange(0,38,2))
plt.yticks(np.arange(0,10,1))

ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.2))
ax.tick_params(which='minor')

ax.grid(visible=True)
ax.legend()
ax.set_title('Vestas v164-8.0')
