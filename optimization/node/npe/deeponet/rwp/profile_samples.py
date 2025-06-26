from profile_generators import *
import matplotlib.pyplot as plt


max_height = 150
vis_grid = jnp.linspace(1, max_height, 151)

plt.figure(figsize=(6, 3.2))
for i in range(8):
    p = N_profile_generator(vis_grid, random.PRNGKey(170393+i+5))
    plt.plot(p + 30*i + 2*vis_grid/6371000*1E6, vis_grid)
plt.ylabel('Height (m)')
plt.xlabel('M-profile')
plt.ylim([vis_grid[0], vis_grid[-1]])
plt.tight_layout()
plt.show()