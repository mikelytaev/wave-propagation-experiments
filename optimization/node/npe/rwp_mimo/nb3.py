import optax
from flax import nnx
from jax import numpy as jnp, lax
from jax import random

from experimental.rwp_jax import PiecewiseLinearNProfileModel, EvaporationDuctModel
from experiments.optimization.node.npe.rwp_mimo.common import MultiAngleRWPModel, add_noise

import jax
import matplotlib.pyplot as plt

from experiments.optimization.node.npe.rwp_mimo.deeponet_inverse import DeepONet, learn_inverse_G

jax.config.update("jax_enable_x64", True)

freq_hz = 3E9
max_range_m = 10000
measure_points_z = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
angles_deg = [0.0]
src_height_m = 50
model = MultiAngleRWPModel(
    measure_points_x=[-1],
    measure_points_z=measure_points_z,
    freq_hz=freq_hz,
    angles_deg=angles_deg,
    max_range_m=max_range_m,
    src_height_m=src_height_m
)


class Proxy:

    def __init__(self, model: MultiAngleRWPModel, z_grid: jnp.ndarray, key=random.PRNGKey(17031993)):
        self.model = model
        self.z_grid = z_grid
        self.key = key

    def __call__(self, N_vals):
        p = PiecewiseLinearNProfileModel(
            self.z_grid,
            N_vals
        )
        self.model.set_N_profile(p)
        v = self.model.compute()
        self.key = random.split(key, 1)[0]
        v = add_noise(v, 30, self.key)[0]
        return jnp.concatenate((v.real, v.imag))


max_height = 150
grid = jnp.linspace(0, max_height, 151)

def surface_duct_N_profile_generator(key=random.PRNGKey(17031993)):
    keys = random.split(key, 2)
    h = jax.random.uniform(keys[0], minval=20, maxval=150)
    p = PiecewiseLinearNProfileModel(
        jnp.array([0, h, h + 20]),
        jnp.array([jax.random.uniform(keys[1], minval=0.0, maxval=30), 0, 0]),
    )
    return p(grid)


def trilinear_duct_N_profile_generator(key=random.PRNGKey(17031993)):
    keys = random.split(key, 5)
    z1 = jax.random.uniform(keys[0], minval=30, maxval=70)
    z2 = jax.random.uniform(keys[1], minval=75, maxval=125)
    n1 = jax.random.uniform(keys[2], minval=15, maxval=25)
    n2mn1 = jax.random.uniform(keys[3], minval=1, maxval=30)
    n0 = jax.random.uniform(keys[4], minval=-5, maxval=20)
    p = PiecewiseLinearNProfileModel(
        jnp.array([0, z1, z2, z2 + 20]),
        jnp.array([n0, n1, n1 - n2mn1, n1 - n2mn1]) - (n1 - n2mn1),
    )
    return p(grid)


def evaporation_duct_N_profile_generator(key=random.PRNGKey(17031993)):
    key = random.split(key, 1)
    edm = EvaporationDuctModel(height_m=jax.random.uniform(key[0], minval=5, maxval=45))
    return edm(grid)


def N_profile_generator(key=random.PRNGKey(17031993)):
    keys = random.split(key, 2)
    selector = jax.random.randint(keys[0], 1, minval=0, maxval=3)[0]
    return lax.cond(
        selector == 0,
        lambda: evaporation_duct_N_profile_generator(keys[1]),
        lambda: lax.cond(
            selector == 1,
            lambda: surface_duct_N_profile_generator(keys[1]),
            lambda: trilinear_duct_N_profile_generator(keys[1])
        )
    )


proxy = Proxy(model, grid)

key = random.PRNGKey(345345)
m = []
for i in range(100):
    key = random.split(key, 1)[0]
    N_vals = N_profile_generator(key)
    m += [proxy(N_vals)]

mean = jnp.mean(jnp.array(m), axis=0)
var = jnp.var(jnp.array(m), axis=0)

inv_G_model = DeepONet(
    rngs=nnx.Rngs(1703),
    samples_num=proxy(N_profile_generator()).shape[0],
    interact_size=100,
    branch_scale=(mean, var),
    trunk_scale=(max_height/2, max_height/2)
)
G_inv, losses = learn_inverse_G(
    proxy,
    N_profile_generator,
    inv_G_model,
    grid,
    max_epoch_num=1000,
    batch_size=25,
    tx = optax.adam(learning_rate=0.002, b1=0.9)
)

plt.plot(jnp.log10(jnp.array(losses)))
plt.grid(True)
plt.show()


model2 = MultiAngleRWPModel(
    measure_points_x=[-1],
    measure_points_z=measure_points_z,
    freq_hz=freq_hz+0.0001,
    angles_deg=angles_deg,
    max_range_m=max_range_m,
    src_height_m=src_height_m
)
proxy2 = Proxy(model2, grid)

vis_grid = jnp.linspace(1, max_height, 151)
plt.figure(figsize=(6, 3.2))
p = trilinear_duct_N_profile_generator(random.PRNGKey(176666567))
p_inv = G_inv(proxy2(p), grid)
plt.plot(model2.env.M_profile(vis_grid), vis_grid)
proxy2(p_inv)
plt.plot(model2.env.M_profile(vis_grid), vis_grid)

p = evaporation_duct_N_profile_generator(random.PRNGKey(176666567))
p_inv = G_inv(proxy2(p), grid)
plt.plot(model2.env.M_profile(vis_grid) + 20, vis_grid)
proxy2(p_inv)
plt.plot(model2.env.M_profile(vis_grid) + 20, vis_grid)

p = surface_duct_N_profile_generator(random.PRNGKey(1677777))
p_inv = G_inv(proxy2(p), grid)
plt.plot(model2.env.M_profile(vis_grid) + 40, vis_grid)
proxy2(p_inv)
plt.plot(model2.env.M_profile(vis_grid) + 40, vis_grid)

plt.show()

# for i  in range(10):
#     p = trilinear_duct_N_profile_generator(random.PRNGKey(i))
#     proxy2(p)
#     plt.plot(model2.env.M_profile(vis_grid), vis_grid)
#
# plt.show()