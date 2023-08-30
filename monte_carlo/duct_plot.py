import numpy as np
from scipy.stats import norm

from rwp.environment import surface_duct
import matplotlib.pyplot as plt
import math as fm


expected_height = 100
expected_strength=20
n = 10
m_0 = 320


plt.figure(figsize=(2, 3.2))

heights = np.linspace(80, 120, 500)
dz = heights[1] - heights[0]
z_grid_m = np.linspace(0, 300, 100)
m_profile = surface_duct(height_m=expected_height, strength=expected_strength, z_grid_m=z_grid_m, m_0=m_0)
max_prop = 0
for height in heights:
    max_prop = max(max_prop, norm.cdf(height+dz, expected_height, 5) - norm.cdf(height, expected_height, 5))

m_min, m_max = min(m_profile), max(m_profile)

pic = np.zeros((200, len(z_grid_m)))
for height in heights:
    v = norm.cdf(height+dz, expected_height, 5) - norm.cdf(height, expected_height, 5)
    m_profile = surface_duct(height_m=height, strength=expected_strength, z_grid_m=z_grid_m, m_0=m_0)
    for z_i in range(0, len(z_grid_m)):
        x_i = round(4*(m_profile[z_i] - m_min))
        pic[x_i, z_i] = v / max_prop

plt.figure(figsize=(2, 3.2))
plt.imshow(pic.T[::-1, :], cmap=plt.get_cmap('binary'))
plt.grid(True)
plt.show()
