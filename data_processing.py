import pandas as pd
import numpy as np
import os
import sys
# Add modules folder to path
os.chdir(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.abspath('modules')) 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import island_plt as ip
import datetime
ip.set_plot_options()
from scipy import integrate
from scipy import interpolate
from matplotlib.ticker import (MultipleLocator)

#%% Choose lot

lotn = 2
lot  = 'Lot ' + str(lotn)
h = '140m' # Chosen height

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

    # Get the file(s) in the folder which are files and have "WindStatus" in the name
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
                          sep = ';',
                          index_col = 0,
                          comment = '#',
                          )

cut_wind_ninja = pd.DataFrame(0, index=np.arange(len(wind_data.index)), columns=['time','wind_speed'])
cut_wind_ninja.set_index("time", inplace = True)
cut_wind_ninja.index = pd.to_datetime(wind_data.index)

# Split Renewable Ninja into DEA timeframe
cut_wind_ninja['wind_speed'] = full_wind_ninja[0:len(wind_data)]['wind_speed'].values

#%% Linear interpolation

# wind_speeds = wind_data['120m']
# for i in range(len(wind_speeds)):
#     if np.isnan(wind_speeds[i]):

# #Fill Nan with mean value
wind_data['120m'] = wind_data['120m'].replace(np.nan,wind_data['120m'].mean())
wind_data['150m'] = wind_data['150m'].replace(np.nan,wind_data['150m'].mean())

x1 = 120
y1 = wind_data['120m']
x2 = 150
y2 = wind_data['150m']

num = ""
for c in h:
    if c.isdigit():
        num = num + c
x = int(num)        

wind_data[h] = y1+(x-x1)*((y2-y1)/(x2-x1))

#%% Cross validate
from sklearn.metrics import mean_squared_error

data1 = wind_data[h].copy()
data2 = cut_wind_ninja.copy()

step_sizes = np.arange(2000,int(len(np.ceil(wind_data[h])/2)))

# step_sizes = [24,168,672]
mse = []
coef = []

for step_size in step_sizes:
    print("Chunk size = " + str(step_size))
    chunks_dea   = [data1[x:x+step_size] for x in range(0, len(data1), step_size)]
    chunks_ninja = [data2[x:x+step_size] for x in range(0, len(data2), step_size)]
    
    for i in range(len(chunks_dea)):
        filtered_dea = data1.copy().drop(chunks_dea[i].index)
        filtered_ninja = data2.copy().drop(chunks_ninja[i].index)
        
        coef.append(chunks_dea[i].mean()/chunks_ninja[i].mean())
        
        predict = filtered_ninja * coef[i]
        
        mse.append(mean_squared_error(filtered_dea, predict))
    
coef_opt = coef[mse.index(min(mse))]
print("Coef_opt = " + str(coef_opt))
print("MSE index = " + str(mse.index(min(mse))) + ", MSE = " + str(min(mse)))

#%% Plot time-series
fig, ax = plt.subplots(1,1,figsize = (10, 5), dpi = 300)

ax.plot(wind_data[h], label = 'DEA (LiDAR buoy)')
ax.plot(cut_wind_ninja,   label = 'Renewables.Ninja')

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
                str('DEA:\n'
                    + 'Std = ' + str(round(wind_data[h].std(), 1)) +'m/s\n'
                    + 'Min = ' + str(round(wind_data[h].min(), 1)) +'m/s\n'
                    + 'Mean = '+ str(round(wind_data[h].mean(),1)) +'m/s\n'
                    + 'Max = ' + str(round(wind_data[h].max(), 1)) +'m/s\n'),
                ha='left', va='top', fontsize = 11, 
                transform=ax.transAxes,
                )

#%% Mean correction
# full_wind_mean = full_wind_ninja/full_wind_ninja.mean() * wind_data[h].mean()
full_wind_mean =  coef_opt * full_wind_ninja
# full_wind_mean =  1.04 * full_wind_ninja

#Initialize new dataframe for mean_cut time series
cut_wind_ninja_mean = pd.DataFrame(0, index=np.arange(len(wind_data.index)), columns=['time','wind_speed'])
cut_wind_ninja_mean.set_index("time", inplace = True)
cut_wind_ninja_mean.index = pd.to_datetime(wind_data.index)

# Split Renewable Ninja into DEA timeframe
cut_wind_ninja_mean['wind_speed'] = full_wind_mean[0:len(wind_data)]['wind_speed'].values

#%% Mean correction histogram
fig, ax = plt.subplots(1,1,figsize = (10, 5), dpi = 300)

ax.hist([wind_data[h].values,
        cut_wind_ninja['wind_speed'].values, 
        cut_wind_ninja_mean['wind_speed'].values],
        bins = np.arange(0,27,1), 
        rwidth=0.9,
        label = ['DEA',
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
# P = [0,40,100,370,650,895,1150,1500,1850,2375,2900,3525,4150,4875,5600,6350,7100,7580,7800,7920,8000] # kW
# v = np.arange(3,13.5,0.5) # m/s
# vestas = Wind_Turbine(8000,3,13,25,P,v)

# Vestas V164/9500
P = [0,115,249,430,613,900,1226,1600,2030,2570,3123,3784,4444,5170,5900,6600,7299,7960,8601,9080,9272,9410,9500] # kW
v = np.arange(3,14.5,0.5) # m/s
vestas = Wind_Turbine(9500,3,14,25,P,v)

# Renewables.ninja mean corrected
v = full_wind_mean['wind_speed'].values
P = vestas.multi_pc(v)
cf_wind = P/vestas.rated_power
cf_wind = pd.Series(cf_wind)
cf_wind.name = "electricity"
    
cf_wind.to_csv('data/wind/wind_cf.csv')

#%% Plot power curves
x = np.linspace(0,36,100)
fig, ax = plt.subplots(figsize = (10,5), dpi = 300)

ax.plot(x, vestas.single_pc(x)/10**3, label = 'Single turbine')
ax.plot(x, vestas.multi_pc(x)/10**3, label = 'Multiple turbines')

ax.set_xlabel('Wind speed [m/s]')
ax.set_ylabel('Power output [MW]')

ax.set_xlim([0,36])
ax.set_ylim([0,10])

plt.xticks(np.arange(0,38,2))
plt.yticks(np.arange(0,11,1))

ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.2))
ax.tick_params(which='minor')

ax.grid(visible=True)
ax.legend()
ax.set_title('Vestas v164-8.0')
