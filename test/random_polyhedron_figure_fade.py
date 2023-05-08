import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sample_sphere_edges(num_samples):
    # Generate random points on the unit sphere
    points = np.random.normal(size=(num_samples, 3))
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    
    return points

# Generate 100 samples from the edges of a sphere
num_samples = 100
points = sample_sphere_edges(num_samples)

# Calculate the convex hull of the points
hull = ConvexHull(points)

# Visualize the convex hull
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the vertices of the simplices
for i in hull.vertices:
    alpha = 1 - (points[i, 2] + 1) / 2 # Adjust the transparency based on z value
    ax.plot(points[i,0], points[i,1], points[i,2], 'ko', alpha=alpha,
            ms = 2)

# Plot the edges of the simplices
edges = set()
for simplex in hull.simplices:
    for i in range(3):
        edge = (simplex[i], simplex[(i+1)%3])
        if not edge in edges:
            edges.add(edge)
            alpha = 1 - (points[edge[0], 2] + 1) / 2 # Adjust the transparency based on z value
            ax.plot(points[[edge[0],edge[1]],0], points[[edge[0],edge[1]],1], points[[edge[0],edge[1]],2], 
                    'k-', alpha=alpha, linewidth = 0.25)

# Set the plot to be spherical
ax.set_box_aspect([1,1,1])
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
plt.show()
