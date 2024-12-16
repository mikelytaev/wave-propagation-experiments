from common import *
from matplotlib.colors import Normalize
from matplotlib.pyplot import figure

from experimental.rwp_jax import PiecewiseLinearNProfileModel, EmptyNProfileModel
import jax.numpy as jnp
from utils import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

jax.config.update("jax_enable_x64", True)


vis_model = RWPModel(params=RWPComputationalParams(
        max_range_m=50000,
        max_height_m=250,
        dx_m=100,
        dz_m=0.5
    ))
f, ax = plt.subplots(1, 3, figsize=(10, 2.3), constrained_layout=True)
f1 = vis_model.calc_field(profiles[10])
extent = (vis_model.fwd_model.x_output_grid()[0]*1e-3, vis_model.fwd_model.x_output_grid()[-1]*1e-3,
          vis_model.fwd_model.z_output_grid()[0], vis_model.fwd_model.z_output_grid()[-1])
norm = Normalize(-60, -10)
ax[0].set_title("Original M-profile (t=10)")
ax[0].imshow(20*jnp.log10(jnp.abs(f1+1e-16)).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('jet'))
ax[0].grid(True)
ax[0].set_xlabel('Range (km)')
ax[0].set_ylabel('Height (km)')

f1_i = vis_model.calc_field(profiles[39])
norm = Normalize(-60, -10)
ax[1].set_title("Inverted M-profile (t=10)")
ax[1].imshow(20*jnp.log10(jnp.abs(f1_i+1e-16)).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('jet'))
ax[1].grid(True)
ax[1].set_xlabel('Range (km)')
ax[1].set_yticklabels([])

f2 = vis_model.calc_field(EmptyNProfileModel())
norm = Normalize(-60, -10)
ax[2].set_title("Standard atmosphere")
im = ax[2].imshow(20*jnp.log10(jnp.abs(f2+1e-16)).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('jet'))
ax[2].grid(True)
ax[2].set_xlabel('Range (km)')
ax[2].set_yticklabels([])

f.colorbar(im, ax=ax[:], shrink=1.0, location='right')
plt.show()
