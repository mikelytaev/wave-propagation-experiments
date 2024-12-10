from copy import deepcopy
from dataclasses import dataclass
from typing import List

import jax
import numpy as np
import optax
from jax import numpy as jnp, tree_util

from experimental.helmholtz_jax import RationalHelmholtzPropagator
from experimental.rwp_jax import GaussSourceModel, TroposphereModel, ComputationalParams, create_rwp_model, \
    AbstractNProfileModel, PiecewiseLinearNProfileModel

from experiments.optimization.node.objective_functions import bartlett


def adam(model, measure, profile_model, gamma=1E-3, learning_rate=0.002):
    model.apply_N_profile(profile_model)
    tx = optax.adam(learning_rate=learning_rate)

    opt_params = model.env.N_profile.params
    opt_params_0 = deepcopy(opt_params)
    opt_state = tx.init(opt_params)
    loss_grad_fn = jax.value_and_grad(loss)

    best_loss = np.inf
    best_params = None
    best_counter = 0

    for i in range(10000):
        #l0 = loss0(opt_params, model, measure)
        #l1 = loss1(opt_params, model)
        #print(f'l0 = {l0}; l1 = {l1}')
        loss_val, grads = loss_grad_fn(opt_params, model, measure, gamma)
        updates, opt_state = tx.update(grads, opt_state)
        opt_params = optax.apply_updates(opt_params, updates)
        print(f'i = {i}; Loss = {loss_val}')

        if loss_val < best_loss:
            best_loss = loss_val
            best_params = deepcopy(opt_params)
            best_counter = 0
        else:
            best_counter += 1

        if best_counter > 25:
            break

    res_model = deepcopy(profile_model)
    res_model.params = best_params
    start_model = deepcopy(profile_model)
    start_model.params = opt_params_0
    return best_params, opt_params_0


@dataclass
class RWPModel:
    measure_points_x: List[int] = None
    measure_points_z: List[int] = None
    fwd_model: RationalHelmholtzPropagator = None
    src: GaussSourceModel = GaussSourceModel(freq_hz=3E9, height_m=10.0, beam_width_deg=3.0)
    env: TroposphereModel = TroposphereModel()
    params: ComputationalParams = ComputationalParams(
        max_range_m=5000,
        max_height_m=250,
        dx_m=100,
        dz_m=1
    )

    def __post_init__(self):
        self.fwd_model = create_rwp_model(self.src, self.env, self.params)

    def calc_field(self, N_profile: AbstractNProfileModel):
        self.env.N_profile = N_profile
        return self.fwd_model.compute(self.src.aperture(self.fwd_model.z_computational_grid()))

    def get_replica(self):
        f = self.fwd_model.compute(self.src.aperture(self.fwd_model.z_computational_grid()))
        return f[self.measure_points_x, self.measure_points_z]


    def apply_N_profile(self, N_profile: AbstractNProfileModel):
        self.env.N_profile = N_profile
        return self.get_replica()


def Bartlett_loss(val, measure):
    etalon_loss0_v = 1 / bartlett(measure, measure)
    f = jnp.ravel(val)
    return 1 / bartlett(measure, f) - etalon_loss0_v


def add_noise(measure, snr, key=jax.random.PRNGKey(1703)):
    signal_level = jnp.mean(abs(measure) ** 2)
    noise_var = signal_level / (10 ** (snr / 10))
    noise_sigma_r = jnp.sqrt(noise_var / 2)
    r_t = jax.random.normal(key, (len(measure), 2)) * noise_sigma_r
    noise = r_t[:, 0] + 1j * r_t[:, 1]
    measure += noise
    return measure


def hvp(f, primals, tangents):
    return jax.jvp(jax.grad(f), primals, tangents)[1]


def operator(params, model: RWPModel):
    model.env.N_profile.params = params
    return model.get_replica()


def loss0(params, model: RWPModel, measure):
    val = operator(params, model)
    return Bartlett_loss(val, measure)


def loss1(params, model: RWPModel):
    model.env.N_profile.params = params
    z_grid_m = jnp.linspace(0, model.env.max_height_m()+20, 201)
    return jnp.linalg.norm(jnp.diff(model.env.N_profile(z_grid_m))) ** 2


def loss(params, model: RWPModel, measure, gamma):
    return loss0(params, model, measure) + gamma*loss1(params, model)


class PLNPM(PiecewiseLinearNProfileModel):

    @property
    def params(self):
        return self.N_vals

    @params.setter
    def params(self, value):
        self.N_vals = value

    def _tree_flatten(self):
        dynamic = (self.z_grid_m, self.N_vals)
        static = {}
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        return cls(z_grid_m=dynamic[0], N_vals=dynamic[1])


tree_util.register_pytree_node(PLNPM,
                               PLNPM._tree_flatten,
                               PLNPM._tree_unflatten)

surface_duct_N = PiecewiseLinearNProfileModel(jnp.array([0, 50, 100]), jnp.array([20.0, 0, 0]))
elevated_duct_N = PiecewiseLinearNProfileModel(jnp.array([0, 100, 150, 300]), jnp.array([20, 20, 0, 0]))
surface_based_duct_N = PiecewiseLinearNProfileModel(jnp.array([0, 50, 75, 100]), jnp.array([10.0, 30, 0, 0]))
surface_based_duct2_N = PiecewiseLinearNProfileModel(jnp.array([0, 50, 120, 100]), jnp.array([30.0, 30, 0, 0]))
