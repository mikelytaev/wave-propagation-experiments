from typing import List

import jax
from jax import numpy as jnp, random

from experimental.rwp_jax import AbstractNProfileModel, TroposphereModel, RWPComputationalParams, RWPGaussSourceModel, \
    create_rwp_model, PiecewiseLinearNProfileModel, AbstractTerrainModel


class MultiAngleRWPModel:

    def __init__(self,
                 measure_points_x: List[int],
                 measure_points_z: List[int],
                 freq_hz: float,
                 angles_deg: List[float],
                 N_profile: AbstractNProfileModel = None,
                 max_range_m: float = 5000,
                 src_height_m: float = 50,
                 terrain: AbstractTerrainModel = None,
                 beam_width_deg=3.0
                 ):
        self.measure_points_x = measure_points_x
        self.measure_points_z = measure_points_z
        self.src_height_m = src_height_m

        self.env = TroposphereModel(
            terrain=terrain,
        )
        if N_profile is not None:
            self.env.N_profile = N_profile

        params = RWPComputationalParams(
            max_range_m=max_range_m,
            max_height_m=250,
            dx_m=100,
            dz_m=1
        )

        self.fwd_model = []
        self.srcs = []
        for angle in angles_deg:
            src = RWPGaussSourceModel(freq_hz=freq_hz+angle*1e-5, height_m=self.src_height_m,
                                      beam_width_deg=beam_width_deg, elevation_angle_deg=angle)
            self.fwd_model += [create_rwp_model(src, self.env, params)]
            self.srcs += [src]

    def set_N_profile(self, N_profile: AbstractNProfileModel):
        self.env.N_profile = N_profile

    def compute(self):
        result = []
        for ind in range(len(self.fwd_model)):
            model = self.fwd_model[ind]
            src = self.srcs[ind]
            f = model.compute(src.aperture(model.z_computational_grid()))
            result += [f[self.measure_points_x, self.measure_points_z]]

        return result

    def compute_fields(self):
        result = []
        for ind in range(len(self.fwd_model)):
            model = self.fwd_model[ind]
            src = self.srcs[ind]
            f = model.compute(src.aperture(model.z_computational_grid()))
            result += [f]

        return result

    def __call__(self, ssp_params):
        self.env.N_profile.params = ssp_params
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



# def operator(params, model: MultiAngleRWPModel):
#     model.set_params(params)
#     return model.get_replica()
#
# def loss0(params, model: AbstractModel, measure, batch_index):
#     val = operator(params, model)
#     return Bartlett_loss(val[batch_index], measure[batch_index])#jnp.linalg.norm(abs(val[batch_index]) - abs(measure[batch_index]))#
#
#
# def loss1(params, model: AbstractModel):
#     model.set_params(params)
#     z_grid_m = jnp.linspace(0, model.z_max()+20, 201)
#     return jnp.linalg.norm(jnp.diff(model.profile(z_grid_m))) ** 2
#
#
# def loss(params, model: AbstractModel, measure, gamma, batch_index):
#     return loss0(params, model, measure, batch_index) + gamma*loss1(params, model)
#
#
# def adam(model, measure, profile_model, gamma=1E-3, learning_rate=0.002, batch_size=None, stop_criteria=25):
#     model.apply_profile(profile_model)
#     tx = optax.adam(learning_rate=learning_rate)
#
#     opt_params = profile_model.params
#     opt_params_0 = deepcopy(opt_params)
#     opt_state = tx.init(opt_params)
#     loss_grad_fn = jax.value_and_grad(loss)
#
#     best_loss = np.inf
#     best_params = None
#     best_counter = 0
#
#     loss_vals = []
#     if ground_truth_profile is not None:
#         z_grid = jnp.linspace(0, profile_model.max_height_m(), 200)
#         ground_truth_errors = []
#     else:
#         ground_truth_errors = None
#
#     t_profile = deepcopy(profile_model)
#     t = time.time()
#     batch_key = jax.random.PRNGKey(42)
#     for i in range(10000):
#         batch_index = jax.random.permutation(batch_key, len(measure))[0:batch_size]
#         l0 = loss0(opt_params, model, measure, batch_index)
#         l1 = loss1(opt_params, model)
#         print(f'l0 = {l0}; l1 = {gamma*l1}')
#         loss_val, grads = loss_grad_fn(opt_params, model, measure, gamma, batch_index)
#         batch_key = jax.random.split(batch_key, 1)[0]
#         updates, opt_state = tx.update(grads, opt_state)
#         opt_params = optax.apply_updates(opt_params, updates)
#
#         loss_vals += [loss_val]
#         if ground_truth_profile is not None:
#             t_profile.params = opt_params
#             ground_truth_errors += [jnp.linalg.norm(ground_truth_profile(z_grid) - t_profile(z_grid)) / jnp.linalg.norm(ground_truth_profile(z_grid))]
#
#         if i % 25 == 0:
#             print(f'i = {i}; Loss = {loss_val}')
#
#         if loss_val < best_loss:
#             best_loss = loss_val
#             best_params = deepcopy(opt_params)
#             best_counter = 0
#         else:
#             best_counter += 1
#
#         if best_counter > stop_criteria:
#             break
#
#     time_s = time.time() - t
#
#     res_profile = deepcopy(profile_model)
#     res_profile.params = best_params
#     start_profile = deepcopy(profile_model)
#     start_profile.params = opt_params_0
#     return OptResult(
#         loss_vals=loss_vals,
#         start_profile=start_profile,
#         res_profile=res_profile,
#         time_s=time_s,
#         ground_truth_errors=ground_truth_errors
#     )
class Proxy:

    def __init__(self, model: MultiAngleRWPModel, z_grid: jnp.ndarray, key=random.PRNGKey(17031993)):
        self.model = model
        self.z_grid = z_grid
        self.key = key

    def __call__(self, N_vals):
        p = PiecewiseLinearNProfileModel(
            jnp.concatenate((self.z_grid, jnp.array([self.z_grid[-1]+50, self.z_grid[-1]+51]))),
            jnp.concatenate((N_vals, jnp.array([0.0, 0.0])))
        )
        self.model.set_N_profile(p)
        v = self.model.compute()
        self.key = random.split(self.key, 1)[0]
        #v = add_noise(v, 30, self.key)[0]
        v = v[0]
        return jnp.concatenate((v.real, v.imag))

    def calc_field(self, N_vals):
        p = PiecewiseLinearNProfileModel(
            jnp.concatenate((self.z_grid, jnp.array([self.z_grid[-1] + 50, self.z_grid[-1] + 51]))),
            jnp.concatenate((N_vals, jnp.array([0.0, 0.0])))
        )
        self.model.set_N_profile(p)
        return self.model.compute_fields()[0]
