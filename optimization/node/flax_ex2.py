from typing import Sequence
import jax
from flax.linen import zeros_init, ones_init
from jax import random, numpy as jnp, tree_util
from flax import linen as nn
import matplotlib.pyplot as plt
from jax._src.nn.initializers import glorot_normal, glorot_uniform
from matplotlib.colors import Normalize
import math as fm

from experiments.optimization.node.helmholtz_jax import AbstractWaveSpeedModel, PiecewiseLinearWaveSpeedModel, \
    ConstWaveSpeedModel
from experiments.optimization.node.objective_functions import bartlett
from experiments.optimization.node.uwa_jax import GaussSourceModel, UnderwaterEnvironmentModel, UnderwaterLayerModel, \
    ComputationalParams, uwa_forward_task, uwa_get_model


class ExplicitMLP(nn.Module):
    features: Sequence[int]

    def setup(self):
        self.layers = [nn.Dense(feat, kernel_init=glorot_normal()) for feat in self.features]

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
        if i != len(self.layers) - 1:
            x = nn.tanh(x)
        # else:
        #     x = nn.tanh(x) * 10
        return x


class MLPWaveSpeedModel(AbstractWaveSpeedModel):

    def __init__(self, c0: float = 1500.0, params=None):
        self.c0 = c0
        self.mlp = ExplicitMLP(features=[100]*2 + [1])
        self.params = params if params else self.mlp.init(random.key(0), random.uniform(random.key(0), (1,1)))

    def __call__(self, z):
        z = jnp.array(z)
        z = z.reshape(len(z), 1)
        return self.c0 + self.mlp.apply(self.params, z)[:, 0]

    def _tree_flatten(self):
        dynamic = (self.c0, self.params)
        static = {}
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        return cls(c0=dynamic[0], params=dynamic[1])


tree_util.register_pytree_node(MLPWaveSpeedModel,
                               MLPWaveSpeedModel._tree_flatten,
                               MLPWaveSpeedModel._tree_unflatten)


src = GaussSourceModel(freq_hz=500.0, depth_m=50.0, beam_width_deg=10.0)
env_simulated = UnderwaterEnvironmentModel(
    layers=[
        UnderwaterLayerModel(
            height_m=200.0,
            sound_speed_profile_m_s=PiecewiseLinearWaveSpeedModel(
                z_grid_m=jnp.array([0.0, 100, 200.0]),
                sound_speed=jnp.array([1510.0, 1500.0, 1505.0])
            ),
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

params = ComputationalParams(
    max_range_m=2000,
    max_depth_m=250,
    x_output_points=3,
    z_output_points=100,
)

field = uwa_forward_task(src=src, env=env_simulated, params=params)
plt.figure(figsize=(6, 3.2))
plt.imshow(
    20*jnp.log10(jnp.abs(field.field+1e-16)).T,
    norm=Normalize(vmin=-70, vmax=-20),
    aspect='auto',
    extent=[0, field.x_grid[-1], field.z_grid[-1], 0],
    cmap=plt.get_cmap('jet')
)
plt.colorbar()
plt.grid(True)
plt.show()

env_replica = UnderwaterEnvironmentModel(
    layers=[
        UnderwaterLayerModel(
            height_m=200.0,
            sound_speed_profile_m_s=MLPWaveSpeedModel(),
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


training_model = uwa_get_model(src=src, env=env_replica, params=params)

c0 = env_simulated.layers[0].sound_speed_profile_m_s(src.depth_m)
k0 = 2 * fm.pi * src.freq_hz / c0
init = src.aperture(k0, training_model.z_computational_grid())

measure = field.field[-1,:]


#@jax.value_and_grad

etalol_loss0_v = 1 / bartlett(measure[2:390:10], measure[2:390:10])


def loss0(params):
    env_replica.layers[0].sound_speed_profile_m_s.params = params
    f = training_model.compute(init)[-1, :]
    return 1 / bartlett(measure[2:390:10], f[2:390:10]) - etalol_loss0_v


@jax.jit
def loss_func(params):
     return loss0(params)


def jac_loss(x):
    return jax.grad(loss_func)(x)


import optax
learning_rate = 0.01
tx = optax.adam(learning_rate=learning_rate)

opt_params = env_replica.layers[0].sound_speed_profile_m_s.params

opt_state = tx.init(opt_params)
loss_grad_fn = jax.value_and_grad(loss_func)

for i in range(301):
    loss_val, grads = loss_grad_fn(opt_params)
    updates, opt_state = tx.update(grads, opt_state)
    opt_params = optax.apply_updates(opt_params, updates)
    if i % 10 == 0:
        env_replica.layers[0].sound_speed_profile_m_s.params = opt_params
        print(f'Loss step {i}: {loss_val}; ssp(1): {env_replica.layers[0].sound_speed_profile_m_s([1.0])}; ssp(100): {env_replica.layers[0].sound_speed_profile_m_s([100.0])}')