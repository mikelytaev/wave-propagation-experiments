import time
from typing import List

import jax
from dataclasses import dataclass

from experimental.helmholtz_jax import RationalHelmholtzPropagator
from experimental.rwp_jax import RWPGaussSourceModel, TroposphereModel, RWPComputationalParams, AbstractNProfileModel, \
    create_rwp_model, EvaporationDuctModel, PiecewiseLinearNProfileModel
import jax.numpy as jnp
from experiments.optimization.node.objective_functions import bartlett, abs_bartlett
from scipy.optimize import minimize
import matplotlib.pyplot as plt


jax.config.update("jax_enable_x64", True)


@dataclass
class RWPModel:
    measure_points_x: List[int] = None
    measure_points_z: List[int] = None
    fwd_model: RationalHelmholtzPropagator = None
    src: RWPGaussSourceModel = RWPGaussSourceModel(freq_hz=3E9, height_m=10.0, beam_width_deg=3.0)
    env: TroposphereModel = TroposphereModel()
    params: RWPComputationalParams = RWPComputationalParams(
        max_range_m=50000,
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
    #seed = int(jnp.linalg.norm(vals**10))
    #print(f'{seed}')
    #measure = add_noise(measure, 30, jax.random.key(seed))
    return pwl_loss0(vals, z_grid, model, measure) + gamma*pwl_loss1(vals)


def realtime(model: RWPModel, true_profiles: List[AbstractNProfileModel], snr=30, gamma=1.0E-3):
    jac = jax.grad(loss)
    z_grid = jnp.linspace(0, true_profiles[0].max_height_m(), 21)
    inverted_profiles = [PiecewiseLinearNProfileModel(z_grid, true_profiles[0](z_grid))]
    inversion_time = [0.0]
    nfev_list = [0]
    for ind, true_profile in enumerate(true_profiles, start=1):
        measure = model.apply_N_profile(true_profile)
        measure = add_noise(measure, snr)

        t = time.time()
        m = minimize(
            method='L-BFGS-B',
            fun=loss,
            args=(z_grid, model, measure, gamma),
            x0=inverted_profiles[-1](z_grid)[:-1:],
            jac=jac,
            options={
                'maxfun': 150,
            },
            #callback=lambda xk: print(
            #    f'{pwl_loss0(xk, z_grid, model, measure)} {gamma * pwl_loss1(xk)}'),
        )
        inversion_time += [time.time() - t]
        nfev_list += [m.nfev]
        inverted_profiles += [PiecewiseLinearNProfileModel(z_grid, jnp.concat((m.x, jnp.array([0.0]))))]
        print(f'Step: {ind}/{len(true_profiles)}')

    t = time.time()
    loss(inverted_profiles[1](z_grid)[:-1:], z_grid, model, measure, gamma)
    fwd_time = time.time() - t

    t = time.time()
    jac(inverted_profiles[1](z_grid)[:-1:], z_grid, model, measure, gamma)
    jac_time = time.time() - t

    t = time.time()
    jax.value_and_grad(loss)(inverted_profiles[1](z_grid)[:-1:], z_grid, model, measure, gamma)
    vg_time = time.time() - t

    print(f'fwd_time: {fwd_time} s; jac_time: {jac_time} s; vg_time: {vg_time}')

    return inverted_profiles, inversion_time, nfev_list


def get_rel_errors(orig_profiles: List[AbstractNProfileModel], inverted_profiles: List[AbstractNProfileModel], z_grid):
    res = []
    for ind in range(len(orig_profiles)):
        res += [
            jnp.linalg.norm((orig_profiles[ind](z_grid)) - (inverted_profiles[ind](z_grid))) / jnp.linalg.norm(
                (orig_profiles[ind](z_grid)+300))]
    print_mean_error(res)
    return res


def plot_rel_error(profiles, inverted_profiles, z_grid):
    errors = []
    for ind in range(0, len(profiles)):
        errors += [jnp.linalg.norm((profiles[ind](z_grid)) - (inverted_profiles[ind](z_grid))) / jnp.linalg.norm(
            (profiles[ind](z_grid)))]

    plt.figure(figsize=(6, 3.2))
    plt.xlabel("Time step number")
    plt.ylabel("Rel. error")
    plt.plot(errors)
    plt.grid(True)
    plt.xlim([0, len(profiles) - 1])
    plt.ylim([0.0, 0.2])
    plt.tight_layout()
    plt.show()

def plot_field_2d(model: RWPModel, N_profile: AbstractNProfileModel):
    f = model.calc_field(N_profile)
    plt.figure(figsize=(6, 3.2))
    extent = (model.fwd_model.x_output_grid()[0], model.fwd_model.x_output_grid()[-1], model.fwd_model.z_output_grid()[0], model.fwd_model.z_output_grid()[-1])
    plt.imshow(20*jnp.log10(jnp.abs(f+1e-16)).T[::-1,:], extent=extent, aspect='auto', cmap=plt.get_cmap('jet'))
    plt.grid(True)
    plt.xlabel('Range (km)')
    plt.show()

def print_mean_error(errors):
    m = jnp.mean(jnp.array(errors))
    s = jnp.std(jnp.array(errors))
    print(f'{m}±{s}')