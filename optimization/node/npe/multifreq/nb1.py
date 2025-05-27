from copy import deepcopy
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import optax
from jax import numpy as jnp
import time

from experimental.helmholtz_jax import RationalHelmholtzPropagator, AbstractWaveSpeedModel, \
    PiecewiseLinearWaveSpeedModel, ConstWaveSpeedModel, LinearSlopeWaveSpeedModel
from experimental.uwa_jax import UWAGaussSourceModel, UWAComputationalParams, uwa_get_model, UnderwaterEnvironmentModel, \
    UnderwaterLayerModel
from experiments.optimization.node.flax.utils import MLPWaveSpeedModel
from experiments.optimization.node.npe.common import Bartlett_loss

from experiments.optimization.node.objective_functions import bartlett
from uwa.environment import munk_profile
import jax
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


class MultiFreqUWAModel:

    def __init__(self,
                 measure_points_x: List[int],
                 measure_points_z: List[int],
                 freqs_hz: List[float],
                 ssp: AbstractWaveSpeedModel = None,
                 max_range_m: float = 5000,
                 src_depth_m: float = 50,
                 ):
        self.measure_points_x = measure_points_x
        self.measure_points_z = measure_points_z
        self.src_depth_m = src_depth_m

        self.env = UnderwaterEnvironmentModel(
            layers = [
                UnderwaterLayerModel(
                    height_m=200.0,
                    sound_speed_profile_m_s=ssp if ssp is not None else LinearSlopeWaveSpeedModel(c0=1500, slope_degrees=0),
                    density=1.0,
                    attenuation_dm_lambda=0.0
                ),
                UnderwaterLayerModel(
                    height_m=jnp.inf,
                    sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1700.0),
                    density=1.5,
                    attenuation_dm_lambda=0.0
                )
            ]
        )

        params = UWAComputationalParams(
            max_range_m=max_range_m,
            max_depth_m=250,
            dx_m=100,
            dz_m=1
        )

        self.fwd_model = []
        self.srcs = []
        for freq in freqs_hz:
            src = UWAGaussSourceModel(freq_hz=freq, depth_m=self.src_depth_m, beam_width_deg=10.0)
            self.fwd_model += [uwa_get_model(src, self.env, params)]
            self.srcs += [src]

    def set_ssp(self, ssp: AbstractWaveSpeedModel):
        self.env.layers[0].sound_speed_profile_m_s = ssp

    def compute(self):
        result = []
        for ind in range(len(self.fwd_model)):
            model = self.fwd_model[ind]
            src = self.srcs[ind]
            c0 = self.env.layers[0].sound_speed_profile_m_s(jnp.array([self.src_depth_m]))[0]
            k0 = 2 * jnp.pi * model.freq_hz / c0
            f = model.compute(src.aperture(k0, model.z_computational_grid()))
            result += [f[self.measure_points_x, self.measure_points_z]]

        return result

    def __call__(self, ssp_params):
        self.env.layers[0].sound_speed_profile_m_s.params = ssp_params
        return self.compute()


def add_noise(measure, snr, key=jax.random.PRNGKey(1703)):
    res = []
    for m in measure:
        signal_level = jnp.mean(abs(m) ** 2)
        noise_var = signal_level / (10 ** (snr / 10))
        noise_sigma_r = jnp.sqrt(noise_var / 2)
        r_t = jax.random.normal(key, (len(m), 2)) * noise_sigma_r
        noise = r_t[:, 0] + 1j * r_t[:, 1]
        m += noise
        res += [m]
    return res


def loss0(params, model: MultiFreqUWAModel, measure):
    res = 0.0
    for ind in range(len(measure)):
        res += Bartlett_loss(model(params)[ind], measure[ind])
    return res


def loss1(ssp_params, model: MultiFreqUWAModel):
    model.env.layers[0].sound_speed_profile_m_s.params = ssp_params
    z_grid_m = jnp.linspace(0, 200+20, 201)
    return jnp.linalg.norm(jnp.diff(model.env.layers[0].sound_speed_profile_m_s(z_grid_m))) ** 2


def loss(params, model: MultiFreqUWAModel, measure, gamma):
    return loss0(params, model, measure) + gamma*loss1(params, model)


@dataclass
class OptResult:
    res_profile: AbstractWaveSpeedModel
    start_profile: AbstractWaveSpeedModel
    loss_vals: Sequence[float]
    time_s: float
    ground_truth_errors: Sequence[float] = None


def adam(model: MultiFreqUWAModel, measure, profile_model: AbstractWaveSpeedModel,
         gamma=1E-3, learning_rate=0.002, batch_size=None, stop_criteria=25,
         ground_truth_profile: AbstractWaveSpeedModel=None):
    model.set_ssp(profile_model)
    tx = optax.adam(learning_rate=learning_rate)

    opt_params = profile_model.params
    opt_params_0 = deepcopy(opt_params)
    opt_state = tx.init(opt_params)
    loss_grad_fn = jax.value_and_grad(loss)

    best_loss = np.inf
    best_params = None
    best_counter = 0

    loss_vals = []
    if ground_truth_profile is not None:
        z_grid = jnp.linspace(0, model.env.layers[0].height_m, 200)
        ground_truth_errors = []
    else:
        ground_truth_errors = None

    t_profile = deepcopy(profile_model)
    t = time.time()
    batch_key = jax.random.PRNGKey(42)
    for i in range(10000):
        batch_index = jax.random.permutation(batch_key, len(measure))[0:batch_size]
        l0 = loss0(opt_params, model, measure)
        l1 = loss1(opt_params, model)
        print(f'l0 = {l0}; l1 = {gamma*l1}')
        loss_val, grads = loss_grad_fn(opt_params, model, measure, gamma)
        batch_key = jax.random.split(batch_key, 1)[0]
        updates, opt_state = tx.update(grads, opt_state)
        opt_params = optax.apply_updates(opt_params, updates)

        loss_vals += [loss_val]
        if ground_truth_profile is not None:
            t_profile.params = opt_params
            ground_truth_errors += [jnp.linalg.norm(ground_truth_profile(z_grid) - t_profile(z_grid)) / jnp.linalg.norm(ground_truth_profile(z_grid))]

        if i % 25 == 0:
            print(f'i = {i}; Loss = {loss_val}')

        if loss_val < best_loss:
            best_loss = loss_val
            best_params = deepcopy(opt_params)
            best_counter = 0
        else:
            best_counter += 1

        if best_counter > stop_criteria:
            break

    time_s = time.time() - t

    res_profile = deepcopy(profile_model)
    res_profile.params = best_params
    start_profile = deepcopy(profile_model)
    start_profile.params = opt_params_0
    return OptResult(
        loss_vals=loss_vals,
        start_profile=start_profile,
        res_profile=res_profile,
        time_s=time_s,
        ground_truth_errors=ground_truth_errors
    )


ssp_1 = PiecewiseLinearWaveSpeedModel(
    z_grid_m=jnp.array([0.0, 75, 200]),
    sound_speed=jnp.array([1510, (1500+1500.0)/2, 1510])
)
freqs_hz = [200, 400]
max_range_m = 2000
measure_points_z = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]
measure_model = MultiFreqUWAModel(
    measure_points_x=[-1],
    measure_points_z=measure_points_z,
    freqs_hz=freqs_hz,
    ssp=ssp_1,
    max_range_m=max_range_m
)

measure = add_noise(measure_model.compute(), 30)


wave_speed_model = MLPWaveSpeedModel(layers=[10]*4, z_max_m=200.0, c0=1510)
model = MultiFreqUWAModel(
    measure_points_x=[-1],
    measure_points_z=measure_points_z,
    freqs_hz=freqs_hz,
    max_range_m=max_range_m
)
ssp1_opt_res = adam(model, measure, learning_rate=0.05, gamma=1E-2*0, profile_model=wave_speed_model)

z_grid_o = jnp.linspace(0, 200, 250)
f, ax = plt.subplots(1, 4, figsize=(6, 3.4), constrained_layout=True)
ax[0].plot(ssp_1(z_grid_o)[::-1], z_grid_o[::-1], color='blue', label="Original")
ax[0].plot(ssp1_opt_res.res_profile(z_grid_o), z_grid_o, color='red', label="Inverted")
ax[0].set_xlabel("SSP (m/s)")
ax[0].set_ylabel("Depth (m)")
ax[0].set_ylim([z_grid_o[-1], z_grid_o[0]])
ax[0].grid(True)

plt.show()