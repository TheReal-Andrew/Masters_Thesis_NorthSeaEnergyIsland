import matplotlib.pyplot as plt
import numpy as np
import island_plt as ip
ip.set_plot_options()

x = [2,3,4,5,6]

ynt1 = [1, 1.3,  1.7,  2.1,    4.0] # Days
ynt2 = [1, 1.4,  4.7, 22.0,   90.0] # Week
ynt3 = [1, 2.3, 12.4, 55.4, 3267.4] # Month

yt1 = [ 2.48,  3.34 ,   4.27,   5.33 ,     9.91] # Days
yt2 = [ 3.73,  5.17 ,  17.52,  82.21 ,   337.88] # Week
yt3 = [16.55, 38.24 , 205.37, 917.63 , 54075.66] # Month

#%% Plot time-series
fig, ax = plt.subplots(1,1,figsize = (10, 5), dpi = 300)

ax.plot(x,ynt1, label = '24 hr-snapshots (Day)')
ax.plot(x,ynt2, label = '168 hr-snapshots (Week)')
ax.plot(x,ynt3, label = '672 hr-snapshots (Month)')

ax.set_ylim([1,10000])
ax.set_xlim([2,6])
ax.set_xticks(np.arange(2,7))

ax.set_yscale('log')

ax.set_xlabel('Number of MAA-variables [-]')
ax.set_ylabel('Normalized time [-]')
ax.set_title('Compute time as a function of number of MAA-variables')

ax.legend(loc = 'upper left', fontsize = 11)
ax.grid(linewidth=1)