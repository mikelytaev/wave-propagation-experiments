import jax

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from experimental.rwp_jax import GaussSourceModel, ComputationalParams, PiecewiseLinearNProfileModel
import jax.numpy as jnp

from experiments.optimization.node.flax.utils import MLPNProfileModel
from experiments.optimization.node.npe.common import adam, RWPModel, add_noise, surface_based_duct_N, surface_duct_N, \
    elevated_duct_N, surface_based_duct2_N

jax.config.update("jax_enable_x64", True)

model = RWPModel(params=ComputationalParams(
        max_range_m=5000,
        max_height_m=250,
        dx_m=100,
        dz_m=1
    ),
    measure_points_x=[-1]*6,
    measure_points_z=[2, 20, 40, 60, 80, 100],
)

measure_surface_duct = model.apply_N_profile(surface_duct_N)
measure_surface_duct = add_noise(measure_surface_duct, 30)
measure_elevated_duct = model.apply_N_profile(elevated_duct_N)
measure_elevated_duct = add_noise(measure_elevated_duct, 30)
measure_surface_based_duct = model.apply_N_profile(surface_based_duct_N)
measure_surface_based_duct = add_noise(measure_surface_based_duct, 30)
measure_surface_based_duct2 = model.apply_N_profile(surface_based_duct2_N)
measure_surface_based_duct2 = add_noise(measure_surface_based_duct2, 30)

layers = [50]*4
best_params, opt_params_0 = adam(model, measure_surface_duct, profile_model=MLPNProfileModel(layers=layers))

z_grid_o = jnp.linspace(0, 250, 250)
model.apply_N_profile(p2)
plt.plot(model.env.M_profile(z_grid_o), z_grid_o)
model.apply_N_profile(MLPNProfileModel(layers=layers))
model.env.N_profile.params = opt_params_0
plt.plot(model.env.M_profile(z_grid_o), z_grid_o)
model.env.N_profile.params = best_params
plt.plot(model.env.M_profile(z_grid_o), z_grid_o)
plt.show()

model.apply_N_profile(surface_based_duct_N)
plt.plot(model.env.M_profile(z_grid_o), z_grid_o)
plt.show()

vis_model = RWPModel(params=ComputationalParams(
        max_range_m=58000,
        max_height_m=250,
        dx_m=100,
        dz_m=1),
    src=GaussSourceModel(freq_hz=3E9, height_m=25.0, beam_width_deg=3.0)
)
f1 = vis_model.calc_field(surface_based_duct_N)
f, ax = plt.subplots(1, 1, figsize=(6, 3.2), constrained_layout=True)
norm = Normalize(-70, -10)
extent = (vis_model.fwd_model.x_output_grid()[0], vis_model.fwd_model.x_output_grid()[-1]*1e-3, vis_model.fwd_model.z_output_grid()[0], vis_model.fwd_model.z_output_grid()[-1])
ax.imshow(20*jnp.log10(abs(f1+1e-16)).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('jet'))
f.tight_layout()
plt.show()