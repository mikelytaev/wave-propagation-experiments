import jax.numpy as jnp
import numpyro

from experimental.helmholtz_jax import LinearSlopeWaveSpeedModel
from experimental.uwa_jax import UWAComputationalParams, UWAGaussSourceModel, UnderwaterEnvironmentModel, \
    uwa_get_model, uwa_forward_task
from experiments.optimization.node.objective_functions import bartlett
import math as fm
import jax
from numpyro.infer import MCMC, SA
import numpyro.distributions as dist

src = UWAGaussSourceModel(freq_hz=50, depth_m=100, beam_width_deg=10)
env = UnderwaterEnvironmentModel(
    sound_speed_profile_m_s=LinearSlopeWaveSpeedModel(c0=1500, slope_degrees=1),
    bottom_profile=lambda x: x*0 + 1000
)
params = UWAComputationalParams(
    max_range_m=50000,
    max_depth_m=500,
    x_output_points=10,
    z_output_points=500,
)

field = uwa_forward_task(src=src, env=env, params=params)
measure = field.field[-1,:]

model = uwa_get_model(src=src, env=env, params=params)
c0 = env.sound_speed_profile_m_s(src.depth_m)
k0 = 2 * fm.pi * src.freq_hz / c0
init = src.aperture(k0, model.z_computational_grid())


def numpyro_model():
    slope_degrees = numpyro.sample("slope_degrees", dist.Uniform(-3.0, 3.0))
    model.wave_speed.slope_degrees = slope_degrees
    f = model.compute(init)
    l = bartlett(measure, f[-1,:])
    sigma = jnp.sqrt(l / len(measure)/2)
    ll = numpyro.sample("ll", dist.Normal(0, sigma), obs=l)


mcmc = MCMC(SA(model=numpyro_model), num_warmup=2000, num_samples=2000, num_chains=1)
mcmc.run(jax.random.PRNGKey(4))
samples = mcmc.get_samples()
mcmc.print_summary()
