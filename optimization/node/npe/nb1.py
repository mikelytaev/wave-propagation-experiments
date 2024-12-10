import time
from typing import List

import jax
from dataclasses import dataclass

import matplotlib.pyplot as plt

from experimental.helmholtz_jax import RationalHelmholtzPropagator
from experimental.rwp_jax import GaussSourceModel, TroposphereModel, ComputationalParams, AbstractNProfileModel, \
    create_rwp_model, PiecewiseLinearNProfileModel
import jax.numpy as jnp

from experiments.optimization.node.objective_functions import bartlett
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)


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


def pwl_operator(z_grid: jnp.ndarray, vals: jnp.ndarray, model: RWPModel):
    return model.apply_N_profile(PiecewiseLinearNProfileModel(z_grid, jnp.concat((vals, jnp.array([0.0])))))


def pwl_loss0(vals: jnp.ndarray, z_grid: jnp.ndarray, model: RWPModel, measure):
    val = pwl_operator(z_grid, vals, model)
    return Bartlett_loss(val, measure)


def pwl_loss1(vals: jnp.ndarray):
    return (jnp.linalg.norm(jnp.diff(jnp.concat((vals, jnp.array([0.0]))))))**2


def loss(vals: jnp.ndarray, z_grid: jnp.ndarray, model: RWPModel, measure, gamma):
    return pwl_loss0(vals, z_grid, model, measure) + gamma*pwl_loss1(vals)


model = RWPModel(params=ComputationalParams(
        max_range_m=7000,
        max_height_m=250,
        dx_m=100,
        dz_m=1
    ),
    measure_points_x=[-1]*11,
    measure_points_z=[2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
)
p2 = PiecewiseLinearNProfileModel(jnp.array([0, 50, 75, 100]), jnp.array([10.0, 30, 0, 0]))
measure = model.apply_N_profile(p2)
jac = jax.grad(loss)
gamma = 1E-3
z_grid = jnp.linspace(0, p2.max_height_m(), 21)
p2_x0 = PiecewiseLinearNProfileModel(jnp.array([0, 20, 80, 100]), jnp.array([0.0, 20, 0, 0]))
x0 = p2_x0(z_grid)[:-1:]
x0_profile = PiecewiseLinearNProfileModel(z_grid, jnp.concat((x0, jnp.array([0.0]))))
m = minimize(
            method='L-BFGS-B',
            fun=loss,
            args=(z_grid, model, measure, gamma),
            x0=x0,
            jac=jac,
            options={
                'maxfun': 150,
            },
        )
inverted_profile = PiecewiseLinearNProfileModel(z_grid, jnp.concat((m.x, jnp.array([0.0]))))

z_grid_o = jnp.linspace(0, 250, 250)
plt.plot(p2(z_grid_o), z_grid_o)
plt.plot(inverted_profile(z_grid_o), z_grid_o)
plt.plot(x0_profile(z_grid_o), z_grid_o)
plt.show()