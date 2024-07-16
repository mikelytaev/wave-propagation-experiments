from copy import deepcopy

import blackjax
import jax.numpy as jnp
from matplotlib.colors import Normalize

from experiments.optimization.node.grid_optimizer import get_optimal_grid
from experiments.optimization.node.utils import RationalHelmholtzPropagator, AbstractWaveSpeedModel, \
    LinearWaveSpeedModel
from uwa.field import AcousticPressureField
from uwa.environment import *
import math as fm
import cmath as cm
from scipy.optimize import minimize, differential_evolution
from jax import tree_util
import jax

import matplotlib.pyplot as plt


@dataclass
class ComputationalParams:
    max_range_m: float
    max_depth_m: float = None
    rational_approx_order = (7, 8)
    dx_m: float = None
    dz_m: float = None
    x_output_points: int = None
    z_output_points: int = None
    precision: float = 0.01


class GaussSourceModel:

    def __init__(self, *, freq_hz, depth_m, beam_width_deg, elevation_angle_deg=0, multiplier=1.0):
        self.freq_hz = freq_hz
        self.depth_m = depth_m
        self.beam_width_deg = beam_width_deg
        self.elevation_angle_deg = elevation_angle_deg
        self.multiplier = multiplier

    def aperture(self, k0, z):
        elevation_angle_rad = fm.radians(self.elevation_angle_deg)
        ww = cm.sqrt(2 * cm.log(2)) / (k0 * cm.sin(fm.radians(self.beam_width_deg) / 2))
        return jnp.array(self.multiplier / (cm.sqrt(cm.pi) * ww) * jnp.exp(-1j * k0 * jnp.sin(elevation_angle_rad) * z)
                         * jnp.exp(-((z - self.depth_m) / ww) ** 2), dtype=complex)

    def max_angle_deg(self):
        return self.beam_width_deg + abs(self.elevation_angle_deg)

    def _tree_flatten(self):
        dynamic = (self.depth_m, self.beam_width_deg, self.elevation_angle_deg, self.multiplier)
        static = {
            'freq_hz': self.freq_hz
        }
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        return cls(depth_m=dynamic[0], beam_width_deg=dynamic[1], elevation_angle_deg=dynamic[2], multiplier=dynamic[3], **static)


tree_util.register_pytree_node(GaussSourceModel,
                               GaussSourceModel._tree_flatten,
                               GaussSourceModel._tree_unflatten)


@dataclass
class UnderwaterEnvironmentModel:
    bottom_sound_speed_m_s: float = 1500
    sound_speed_profile_m_s: AbstractWaveSpeedModel = LinearWaveSpeedModel(c0=1500, slope_degrees=0)
    bottom_profile: "function" = lambda x: x*0 + 300
    bottom_density_g_cm: float = 1
    bottom_attenuation_dm_lambda: float = 0.0

    def _tree_flatten(self):
        dynamic = (self.sound_speed_profile_m_s, self.bottom_profile)
        static = {
            'bottom_sound_speed_m_s': self.bottom_sound_speed_m_s,
            'bottom_density_g_cm': self.bottom_density_g_cm,
            'bottom_attenuation_dm_lambda': self.bottom_attenuation_dm_lambda,
        }
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        unf = cls(sound_speed_profile_m_s=dynamic[0], ref_debottom_profilepth=dynamic[1], **static)
        return unf


tree_util.register_pytree_node(UnderwaterEnvironmentModel,
                               UnderwaterEnvironmentModel._tree_flatten,
                               UnderwaterEnvironmentModel._tree_unflatten)


def check_computational_params(params: ComputationalParams):
    if params.x_output_points is None and params.dx_m is None:
        raise ValueError("x output grid (x_output_points or dx_m) is not specified!")
    if params.x_output_points is not None and params.dx_m is not None:
        raise ValueError("only one x output grid parameter (x_output_points or dx_m) should be specified!")

    if params.z_output_points is None and params.dz_m is None:
        raise ValueError("z output grid (z_output_points or dz_m) is not specified!")
    if params.z_output_points is not None and params.dz_m is not None:
        raise ValueError("only one z output grid parameter (z_output_points or dz_m) should be specified!")


def minmax_k(env: UnderwaterEnvironmentModel, params: ComputationalParams):
    k_func = lambda z: 2 * fm.pi * src.freq_hz / env.sound_speed_profile_m_s(z)
    result_ga = differential_evolution(
        func=k_func,
        bounds=[(0, 1000)],
        popsize=30,
        disp=False,
        recombination=1,
        strategy='randtobest1exp',
        tol=1e-5,
        maxiter=10000,
        polish=False
    )
    k_min = result_ga.fun

    k_func = lambda z: -2 * fm.pi * src.freq_hz / env.sound_speed_profile_m_s(z)
    result_ga = differential_evolution(
        func=k_func,
        bounds=[(0, 1000)],
        popsize=30,
        disp=False,
        recombination=1,
        strategy='randtobest1exp',
        tol=1e-5,
        maxiter=10000,
        polish=False
    )
    k_max = -result_ga.fun

    print(f'k_min: {k_min}, k_max: {k_max}')
    return k_min, k_max


def uwa_get_model(src: GaussSourceModel, env: UnderwaterEnvironmentModel, params: ComputationalParams) -> RationalHelmholtzPropagator:
    check_computational_params(params)

    params = deepcopy(params)
    max_angle_deg = src.max_angle_deg()
    c0 = env.sound_speed_profile_m_s(src.depth_m)
    k0 = 2 * fm.pi * src.freq_hz / c0
    kz_max = k0 * fm.sin(fm.radians(max_angle_deg))

    min_res = minimize(lambda x: -env.bottom_profile(x), x0=[0], bounds=[(0, params.max_range_m)])
    max_bottom_height = float(env.bottom_profile(min_res.x[0]))
    if params.max_depth_m:
        params.max_depth_m = max(params.max_depth_m, max_bottom_height*1.1)
    else:
        params.max_depth_m = max_bottom_height * 1.1

    k_min, k_max = minmax_k(env, params)

    if params.x_output_points:
        params.dx_m = params.max_range_m / (params.x_output_points - 1)
    if params.z_output_points:
        params.dz_m = params.max_depth_m / (params.z_output_points - 1)
    beta, dx_computational, dz_computational = get_optimal_grid(
        kz_max, k_min, k_max, params.precision / params.max_range_m,
        dx_max=params.dx_m,
        dz_max=params.dz_m)
    if params.dx_m:
        dx_computational = params.dx_m / fm.ceil(params.dx_m / dx_computational)
    if params.dz_m:
        dz_computational = params.dz_m / fm.ceil(params.dz_m / dx_computational)

    params.max_range_m = fm.ceil(params.max_range_m / dx_computational) * dx_computational
    params.max_depth_m = fm.ceil(params.max_depth_m / dz_computational) * dz_computational

    if not params.x_output_points:
        params.x_output_points = round(params.max_range_m / params.dx_m) + 1
    if not params.z_output_points:
        params.z_output_points = round(params.max_depth_m / params.dz_m) + 1

    x_grid_scale = round(params.dx_m / dx_computational)
    z_grid_scale = round(params.dz_m / dz_computational)
    x_computational_points = params.x_output_points * x_grid_scale
    z_computational_points = params.z_output_points * z_grid_scale

    x_computational_grid = jnp.linspace(0, params.max_range_m, x_computational_points)
    z_computational_grid = jnp.linspace(0, params.max_depth_m, z_computational_points)

    x_output_grid = jnp.linspace(0, params.max_range_m, params.x_output_points)
    z_output_grid = jnp.linspace(0, params.max_depth_m, params.z_output_points)

    print(f'beta: {beta}, dx: {dx_computational}, dz: {dz_computational}')

    model = RationalHelmholtzPropagator(
        beta=beta,
        dx_m=dx_computational,
        dz_m=dz_computational,
        x_n=len(x_computational_grid),
        z_n=len(z_computational_grid),
        x_grid_scale=x_grid_scale,
        z_grid_scale=z_grid_scale,
        order=(7, 8),
        wave_speed=env.sound_speed_profile_m_s,
        freq_hz=src.freq_hz
    )

    return model


def uwa_forward_task(src: GaussSourceModel, env: UnderwaterEnvironmentModel, params: ComputationalParams) -> AcousticPressureField:
    model = uwa_get_model(src, env, params)
    c0 = env.sound_speed_profile_m_s(src.depth_m)
    k0 = 2 * fm.pi * src.freq_hz / c0
    init = src.aperture(k0, model.z_computational_grid())
    f = model.compute(init)
    return AcousticPressureField(freq_hz=src.freq_hz, x_grid=model.x_output_grid(), z_grid=model.z_output_grid(),
                                 field=f)


src = GaussSourceModel(freq_hz=50, depth_m=100, beam_width_deg=10)
env = UnderwaterEnvironmentModel(
    sound_speed_profile_m_s=LinearWaveSpeedModel(c0=1500, slope_degrees=1),
    bottom_profile=lambda x: x*0 + 1000
)
params = ComputationalParams(
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

measure = field.field[-1,:] + 0.00139055*(jax.random.normal(rng_key, field.field[-1,:].shape) + 1j*jax.random.normal(rng_key, field.field[-1,:].shape))

#measures = [field.nearest_value(params.max_range_m, z) for z in np.linspace(10, 300, 30)]

model = uwa_get_model(src=src, env=env, params=params)
c0 = env.sound_speed_profile_m_s(src.depth_m)
k0 = 2 * fm.pi * src.freq_hz / c0
init = src.aperture(k0, model.z_computational_grid())
slopes = np.linspace(-3.0, 3.0, 301)
err = slopes*0


@jax.value_and_grad
def loss(ssp: LinearWaveSpeedModel):
    model.wave_speed = ssp
    f = model.compute(init)
    return jnp.linalg.norm(f[-1, :] - measure)


def log_likelihood(slope_degrees):
    #jax.debug.print("🤯 {x} 🤯", x=slope_degrees)
    model.wave_speed.slope_degrees = slope_degrees
    f = model.compute(init)
    return -len(measure)*jnp.log(jnp.linalg.norm(f[-1, :] - measure))


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


for ind, slope in enumerate(slopes):
    print(ind)
    err[ind] = log_likelihood(slope)


mh_gen = Metropolis_Hastings(log_likelihood, slope_gen)
for ind in range(0, 10000):
    print(ind)
    print(next(mh_gen))


plt.figure(figsize=(6, 3.2))
plt.plot(slopes, err)
plt.grid(True)
plt.show()
