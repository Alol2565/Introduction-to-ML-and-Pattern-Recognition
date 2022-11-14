import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

k = 4
true_location = np.random.multivariate_normal([0, 0], [[0.25, 0], [0, 0.25]], 1)
# plot the true location and circle with radius 1
plt.figure()
plt.plot(true_location[0, 0], true_location[0, 1], 'rx')
circle = plt.Circle((0, 0), 1, color='r', fill=False)
theta = np.linspace(0, 2 * np.pi, k, endpoint=False)
x = np.cos(theta)
y = np.sin(theta)
references = np.vstack((x, y)).T
plt.plot(x, y, 'bo')
plt.gca().add_patch(circle)
plt.axis('equal')
plt.grid()
plt.savefig('true_location.png')

# measurement with white noise and plot the result
std = 0.3 * np.ones(k)
measurement = np.zeros(k)
for i in range(k):
    measurement[i] = np.random.normal(np.linalg.norm(true_location - references[i]), std[i])

rv = multivariate_normal(mean=[0, 0], cov=[[0.25, 0], [0, 0.25]])
x = np.arange(-2, 2, 0.01)
y = np.arange(-2, 2, 0.01)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

def map_function(pos):
    Z = np.zeros((pos.shape[0], pos.shape[1]))
    for i in range(pos.shape[0]):
        for j in range(pos.shape[1]):
            Z[i, j] = np.log(rv.pdf(pos[i, j])) - 0.5 * np.sum((measurement - np.linalg.norm(pos[i, j] - references, axis=1)) ** 2 / (2 * std ** 2))
    return Z

fig = plt.figure(figsize=(6,5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height]) 
cp = plt.contourf(X, Y, map_function(pos))
# plt.clabel(cp, inline=True, fontsize=10)
plt.colorbar(cp)
ax.set_title('Contour Plot')
ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
plt.plot(true_location[0, 0], true_location[0, 1], 'gx')
ind = np.unravel_index(np.argmax(map_function(pos), axis=None), map_function(pos).shape)
plt.plot(x[ind[0]], y[ind[1]], 'rx')
plt.show()