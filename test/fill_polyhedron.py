# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:14:13 2023

@author: lukas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# Generate some random 2D datapoints
np.random.seed(123)
points = np.random.rand(20, 2)

# Find the convex hull of the points
hull = ConvexHull(points)

# Plot the points and the convex hull
plt.plot(points[:,0], points[:,1], 'o')
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

# Fill the area inside the convex hull
hull_poly = plt.Polygon(points[hull.vertices], alpha=0.2)
plt.gca().add_patch(hull_poly)

plt.show()