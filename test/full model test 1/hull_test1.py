# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 20:54:18 2023

@author: lukas
"""

import os
import sys
# Add modules folder to path
sys.path.append(os.path.abspath('../../modules')) 

import numpy as np
import island_plt as ip
ip.set_plot_options()

from scipy.spatial import ConvexHull

from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#%% Data
def sample_spherical(npoints, ndim=3):
    np.random.seed(0)
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

phi = np.linspace(0, np.pi, 20)
theta = np.linspace(0, 2 * np.pi, 40)
x = np.outer(np.sin(theta), np.cos(phi))
y = np.outer(np.sin(theta), np.sin(phi))
z = np.outer(np.cos(theta), np.ones_like(phi))

xi, yi, zi = sample_spherical(8)

p = np.array([xi, yi, zi]).T

#%% Plot
# From https://stackoverflow.com/questions/27270477/3d-convex-hull-from-point-cloud

colors = ['tab:blue', 'tab:red', 'tab:purple']
ax = plt.axes(projection = '3d')

# Points
ax.scatter(xi, yi, zi, s=50, c = colors[1], zorder=10)

# Define Hull & Edhes
hull = ConvexHull(p)
edges = zip(*p)

# # Plot simplex and cycle back
for s in hull.simplices:
    s = np.append(s, s[0])  # Here we cycle back to the first coordinate
    ax.plot(p[s, 0], p[s, 1], p[s, 2], colors[0])
    


#%% Alternative plot

fig = plt.figure()

colors = ['tab:blue', 'tab:red', 'aliceblue']
ax = plt.axes(projection = '3d')

# Get rid of colored axes planes
# First remove fill
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Now set color to white (or whatever is "invisible")
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

ax.xaxis.set_ticklabels(['y'])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])



for line in ax.xaxis.get_ticklines():
    line.set_visible(False)
for line in ax.yaxis.get_ticklines():
    line.set_visible(False)
for line in ax.zaxis.get_ticklines():
    line.set_visible(False)

# Bonus: To get rid of the grid as well:
# ax.grid(False)

# Points
# ax.scatter(xi, yi, zi, s=50, c = colors[1], zorder=10)
ax.plot(xi, yi, zi, 'o', c = colors[1], ms=7)


# Define Hull & Edhes
hull = ConvexHull(p)
edges = zip(*p)

# Plot trisurface  
ls = LightSource(azdeg=225.0, altdeg=45.0)

ss = ax.plot_trisurf(xi, yi, zi, triangles=hull.simplices,
                     alpha=0.8, color = colors[0],
                     edgecolor = colors[2], linewidth = 3)

# plt.colorbar(ss, shrink=0.7)
    
fig.savefig('hull_plot_test1.svg', format = 'pdf', bbox_inches='tight')

