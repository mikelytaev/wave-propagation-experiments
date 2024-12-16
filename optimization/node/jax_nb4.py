import jax.numpy as jnp
import numpyro
from matplotlib.colors import Normalize

from experimental.helmholtz_jax import LinearSlopeWaveSpeedModel
from experimental.uwa_jax import UWAComputationalParams, UWAGaussSourceModel, UnderwaterEnvironmentModel, \
    uwa_get_model, uwa_forward_task
from uwa.environment import *
import math as fm
import jax
from numpyro.infer import MCMC, SA
import numpyro.distributions as dist

import matplotlib.pyplot as plt


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

plt.figure(figsize=(6, 3.2))
plt.imshow(
    20*jnp.log10(jnp.abs(field.field+1e-16)).T,
    norm=Normalize(vmin=-80, vmax=-40),
    aspect='auto',
    extent=[0, field.x_grid[-1], field.z_grid[-1], 0],
    cmap=plt.get_cmap('jet')
)
plt.colorbar()
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 3.2))
plt.plot(field.z_grid, 20*jnp.log10(jnp.abs(field.field[-1,:]+1e-16)))
plt.grid(True)
plt.show()

print(env.sound_speed_profile_m_s(field.z_grid[-1]))

import datetime
rng_key = jax.random.key(datetime.datetime.now().microsecond)

measure = field.field[-1,:]# + 0.0139055*(jax.random.normal(rng_key, field.field[-1,:].shape) + 1j*jax.random.normal(rng_key, field.field[-1,:].shape))

#measures = [field.nearest_value(params.max_range_m, z) for z in np.linspace(10, 300, 30)]

model = uwa_get_model(src=src, env=env, params=params)
c0 = env.sound_speed_profile_m_s(src.depth_m)
k0 = 2 * fm.pi * src.freq_hz / c0
init = src.aperture(k0, model.z_computational_grid())


@jax.value_and_grad
def loss(ssp: LinearSlopeWaveSpeedModel):
    model.wave_speed = ssp
    f = model.compute(init)
    return jnp.linalg.norm(f[-1, :] - measure)


def log_likelihood(c0, slope_degrees):
    #jax.debug.print("🤯 {x} 🤯", x=slope_degrees)
    model.wave_speed.c0 = c0
    model.wave_speed.slope_degrees = slope_degrees
    f = model.compute(init)
    return -len(measure)*jnp.log(jnp.linalg.norm(f[-1, ::50] - measure[::50]))


logdensity = lambda x: log_likelihood(**x)


def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        jax.debug.print("🤯 {x} 🤯", x=state)
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


# initial_position = {"slope_degrees": 0.0}
# from datetime import date
# rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))
# warmup = blackjax.window_adaptation(blackjax.nuts, logdensity, progress_bar=True)
# rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
# (state, parameters), _ = warmup.run(warmup_key, initial_position, num_steps=50)
# inv_mass_matrix = np.array([1.0])
# step_size = 1e-1
# #parameters = {'step_size': step_size, 'inverse_mass_matrix': inv_mass_matrix}
# kernel = blackjax.nuts(logdensity, **parameters).step
# initial_state = blackjax.nuts.init(initial_position, logdensity)
# states = inference_loop(sample_key, kernel, initial_state, 200)
#
# mcmc_samples = states.position
# plt.figure(figsize=(6, 3.2))
# plt.plot(mcmc_samples['slope_degrees'])
# plt.grid(True)
# plt.show()

rng_key = jax.random.split(rng_key, 1)

def slope_gen():
    global rng_key
    rng_key = jax.random.split(rng_key[0], 1)
    yield jax.random.uniform(rng_key, (1,), minval=-3, maxval=3)


def Metropolis_Hastings(log_likelihood, generator):
    global rng_key
    x_i = next(generator())
    l_i = log_likelihood(x_i)
    while True:
        x_j = next(generator())
        l_j = log_likelihood(x_j)
        prop = jnp.exp(l_j - l_i)
        rng_key = jax.random.split(rng_key[0], 1)
        if prop > jax.random.uniform(rng_key, (1,), minval=0, maxval=1):
            x_i, l_i = x_j, l_j
        yield x_i, l_i


slopes = np.linspace(-10.0, 10.0, 3001)
c0s = [1500]#np.linspace(1400, 1650, 10001)
err = np.empty(shape=(len(c0s), len(slopes)), dtype=float)
for ind_c0, c0 in enumerate(c0s):
    print(ind_c0)
    for ind_slopes, slope in enumerate(slopes):
        err[ind_c0, ind_slopes] = log_likelihood(c0, slope)


plt.figure(figsize=(6, 3.2))
#plt.plot(slopes, err)
plt.imshow(err.T, extent=[c0s[0], c0s[-1], slopes[0], slopes[-1]], aspect='auto', norm=Normalize(1000, 4000), cmap=plt.get_cmap('jet'))
#plt.grid(True)
plt.show()


plt.figure(figsize=(6, 3.2))
plt.plot(slopes, err[0,:])
#plt.grid(True)
plt.show()


# mh_gen = Metropolis_Hastings(log_likelihood, slope_gen)
# for ind in range(0, 10000):
#     print(ind)
#     print(next(mh_gen))

# mcmc = MCMC(NUTS(potential_fn=lambda x: -log_likelihood(x)), num_warmup=10, num_samples=10)
# mcmc.run(jax.random.PRNGKey(0), init_params=jnp.array([1.2,]))
# samples = mcmc.get_samples()
# mcmc.print_summary()
#
# plt.figure(figsize=(6, 3.2))
# plt.plot(slopes, err)
# plt.grid(True)
# plt.show()


def numpyro_model():
    slope_degrees = numpyro.sample("slope_degrees", dist.Uniform(0.5, 1.5))
    model.wave_speed.slope_degrees = slope_degrees
    f = model.compute(init)
    l = jnp.linalg.norm(f[-1, :] - measure)
    sigma = jnp.sqrt(l / len(measure)/2)
    ll = numpyro.sample("ll", dist.Normal(0, sigma), obs=l)


mcmc = MCMC(SA(model=numpyro_model), num_warmup=100, num_samples=100, num_chains=1)
mcmc.run(jax.random.PRNGKey(4))
samples = mcmc.get_samples()
mcmc.print_summary()


def numpyro_model_t(slope_degrees):
    model.wave_speed.slope_degrees = slope_degrees
    f = model.compute(init)
    l = jnp.linalg.norm(f[-1, :] - measure)
    sigma = jnp.sqrt(l / len(measure)/2)
    return jax.scipy.stats.norm.pdf(l, 0, sigma)


slopes = np.linspace(-3, 3, 301)
err2 = slopes*0
for ind, slope in enumerate(slopes):
    print(ind)
    err2[ind] = numpyro_model_t(slope)

plt.figure(figsize=(6, 3.2))
plt.plot(slopes, jnp.log10(err2))
plt.grid(True)
plt.show()
