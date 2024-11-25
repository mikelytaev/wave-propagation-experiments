from typing import Sequence

from flax import linen as nn
from jax import random, numpy as jnp
from jax import numpy as jnp, tree_util

from experimental.helmholtz_jax import AbstractWaveSpeedModel


class ExplicitMLP(nn.Module):
    features: Sequence[int]

    def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.features]

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = nn.relu(x)
        return x


class MLPWaveSpeedModel(AbstractWaveSpeedModel):

    def __init__(self, c0: float = 1500.0, z_max_m = 100.0, params=None):
        self.c0 = c0
        self.mlp = ExplicitMLP(features=[500]*10 + [1])
        self.z_max_m = z_max_m
        self.params = params if params else self.mlp.init(random.key(0), jnp.ones((1, 1)))

    def apply(self, params, z):
        z = 2 * jnp.array(z) / self.z_max_m - 1
        z = z.reshape(len(z), 1)
        return self.c0 + self.mlp.apply(params, z)[:,0]

    def __call__(self, z):
        return self.apply(self.params, z)

    def _tree_flatten(self):
        dynamic = (self.c0, self.params)
        static = {'z_max_m': self.z_max_m}
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        return cls(c0=dynamic[0], params=dynamic[1], **static)


tree_util.register_pytree_node(MLPWaveSpeedModel,
                               MLPWaveSpeedModel._tree_flatten,
                               MLPWaveSpeedModel._tree_unflatten)