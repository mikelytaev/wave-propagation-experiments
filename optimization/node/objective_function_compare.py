import jax.numpy as jnp
from matplotlib.colors import Normalize

from experimental.helmholtz_jax import LinearSlopeWaveSpeedModel, StaircaseRhoModel
from experimental.uwa_jax import UWAComputationalParams, UWAGaussSourceModel, UnderwaterEnvironmentModel, \
    uwa_get_model, uwa_forward_task
import math as fm
import jax

import matplotlib.pyplot as plt


src = UWAGaussSourceModel(freq_hz=50, depth_m=100, beam_width_deg=10)
env = UnderwaterEnvironmentModel(
    sound_speed_profile_m_s=LinearSlopeWaveSpeedModel(c0=1500.0, slope_degrees=1.0),
    rho_profile=StaircaseRhoModel(heights=[0, 700.0], vals=[1.0, 1.5]),
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

measure = field.field[-1, 1:-1]# + 0.0139055*(jax.random.normal(rng_key, field.field[-1,:].shape) + 1j*jax.random.normal(rng_key, field.field[-1,:].shape))

model = uwa_get_model(src=src, env=env, params=params)
c0 = env.sound_speed_profile_m_s(src.depth_m)
k0 = 2 * fm.pi * src.freq_hz / c0
init = src.aperture(k0, model.z_computational_grid())


def l2compare(c0, slope_degrees):
    model.wave_speed.c0 = c0
    model.wave_speed.slope_degrees = slope_degrees
    f = model.compute(init)
    return -len(measure)*jnp.log(jnp.linalg.norm(f[-1, 1:-1] - measure[:]))


@jax.jit
def bartlett(c0, slope_degrees):
    model.wave_speed.c0 = c0
    model.wave_speed.slope_degrees = slope_degrees
    f = model.compute(init)[-1, 1:-1]
    w = f / jnp.linalg.norm(f)
    return abs(jnp.dot(w.conj(), measure) * jnp.dot(measure.conj(), w))


k_matrix = jnp.matmul(measure.reshape(len(measure), 1), measure.conj().reshape(1, len(measure)))
k_inv = jnp.linalg.inv(k_matrix)


def mv(c0, slope_degrees):
    model.wave_speed.c0 = c0
    model.wave_speed.slope_degrees = slope_degrees
    f = model.compute(init)[-1, 1:-1]
    w = f / jnp.linalg.norm(f)
    return 1 / abs(w.conj().reshape(1, len(w)) @ k_inv @ w.reshape(len(w), 1))[0, 0]


@jax.jit
def replica(args):
    c0, slope_degrees = args
    model.wave_speed.c0 = c0
    model.wave_speed.slope_degrees = slope_degrees
    return model.compute(init)[-1, 1:-1]


replica_grad = jax.jacfwd(replica)


@jax.jit
def mcm(c0, slope_degrees):
    f = replica((c0, slope_degrees))
    #f2 = replica((c0-0.05, slope_degrees))
    #f3 = replica((c0+0.05, slope_degrees))
    #grad = replica_grad((c0, slope_degrees))
    v = jnp.column_stack((f,))
    #c = jnp.array([1.0, 1.0, 1.0], dtype=complex).reshape(3, 1)
    c = jnp.array([jnp.dot(f.conj(), f)]).reshape(1, 1).conj()
    gain = (c.T.conj() @ jnp.linalg.inv(v.T.conj() @ k_inv @ v) @ c)
    return abs(gain)[0, 0]


#
# slopes = np.linspace(-3.0, 3.0, 301)
# c0s = np.linspace(1350, 1650, 301)
# err = np.empty(shape=(len(c0s), len(slopes)), dtype=float)
# for ind_c0, c0 in enumerate(c0s):
#     print(ind_c0)
#     for ind_slopes, slope in enumerate(slopes):
#         #print(ind_slopes)
#         err[ind_c0, ind_slopes] = mv(c0, slope)
#
#
# plt.figure(figsize=(6, 3.2))
# #plt.plot(slopes, err)
# plt.imshow(jnp.log10(err.T[::-1,:]), extent=[c0s[0], c0s[-1], slopes[0], slopes[-1]], aspect='auto',  cmap=plt.get_cmap('jet'))
# #plt.grid(True)
# plt.show()
#
#
# plt.figure(figsize=(6, 3.2))
# plt.plot(slopes, jnp.log10(err[0,:]))
# #plt.grid(True)
# plt.show()


c0_bounds = [1400.0, 1550.0]
slope_bounds = [-3.0, 3.0]

bounds = [c0_bounds, slope_bounds]
x0 = [1470, 0]


from scipy.optimize import dual_annealing


func = lambda x: 1/float(bartlett(x[0], x[1]))
ret = dual_annealing(
    func,
    bounds=bounds,
    callback=lambda x, f, ctx: print(x)

)
print(ret)
