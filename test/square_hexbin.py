# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:15:02 2023

@author: lukas
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate random data
x = np.random.randn(10000)
y = np.random.randn(10000)

# Create 2D histogram
hist, xedges, yedges = np.histogram2d(x, y, bins=25)

# Create grid for pcolormesh
X, Y = np.meshgrid(xedges, yedges)

# Create pcolormesh plot with square bins
plt.pcolormesh(X, Y, hist.T, cmap='Blues')

# Add a colorbar
plt.colorbar()