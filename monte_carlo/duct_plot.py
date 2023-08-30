import numpy as np
from matplotlib.colors import Normalize
from scipy.stats import norm

from rwp.environment import surface_duct
import matplotlib.pyplot as plt
import math as fm


expected_height = 100
expected_strength=20
n = 10
m_0 = 320


plt.figure(figsize=(2, 3.2))

heights = np.linspace(50, 150, 500)
dz = heights[1] - heights[0]
z_grid_m = np.linspace(0, 300, 1000)
m_profile = surface_duct(height_m=expected_height, strength=expected_strength, z_grid_m=z_grid_m, m_0=m_0)
max_prop = 0
for height in heights:
    max_prop = max(max_prop, norm.cdf(height+dz, expected_height, 10) - norm.cdf(height, expected_height, 10))

m_min, m_max = min(m_profile), max(m_profile)

x_n = 1000
pic = np.zeros((x_n, len(z_grid_m)))+1e-16
for height in heights:
    v = norm.cdf(height+dz, expected_height, 10) - norm.cdf(height, expected_height, 10)
    m_profile = surface_duct(height_m=height, strength=expected_strength, z_grid_m=z_grid_m, m_0=m_0)
    for z_i in range(0, len(z_grid_m)):
        x_i = round(x_n*(m_profile[z_i] - m_min) / (m_max - m_min))
        if x_i >= x_n:
            continue
        pic[x_i, z_i] = max(pic[x_i, z_i], v)

#plt.figure(figsize=(6, 3.2))
extent = [m_min, m_max, 0, z_grid_m[-1]]
plt.imshow((pic.T[::-1, :]), norm=Normalize(0, max_prop), extent=extent, aspect='auto', cmap=plt.get_cmap('binary'))
plt.colorbar(fraction=0.046, pad=0.04)
plt.grid(True)
plt.tight_layout()
plt.show()
