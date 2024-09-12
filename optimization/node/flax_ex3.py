from typing import Sequence
import jax
from jax import random, numpy as jnp
from flax import linen as nn
import matplotlib.pyplot as plt

from experiments.optimization.node.helmholtz_jax import AbstractWaveSpeedModel


def func(x):
    return -jnp.sin(2*(x - 10)) / (x - 10)


x_test_grid = jnp.linspace(-15, 15, 1000).reshape((1000, 1))
plt.figure(figsize=(6, 3.2))
#plt.plot(x_test_grid, model.apply(params, x_test_grid))
plt.plot(x_test_grid, func(x_test_grid))
plt.show()
plt.grid(True)


import optax
learning_rate = 1
tx = optax.adam(learning_rate=learning_rate)


def loss(x):
    return func(x[0])


params = [5.0]
opt_state = tx.init(params)
loss_grad_fn = jax.value_and_grad(loss)

for i in range(301):
    loss_val, grads = loss_grad_fn(params)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    if i % 10 == 0:
        print(f'Loss step {i}: {loss_val}, x = {params[0]}')



tx2 = optax.adam(learning_rate=0.1)


class ExplicitMLP(nn.Module):
    features: Sequence[int]

    def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.features]

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = nn.tanh(x)
        return x


model = ExplicitMLP(features=[100]*2 + [1])
key1, key2 = random.split(random.key(0), 2)
params = model.init(key2, (1.0,))


def loss2(params):
    y = model.apply(params, (1.0,))*5
    return func(y)[0]


opt_state = tx.init(params)
loss_grad_fn = jax.value_and_grad(loss2)

for i in range(301):
  loss_val, grads = loss_grad_fn(params)
  updates, opt_state = tx.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  if i % 10 == 0:
    print(f'Loss step {i}: {loss_val}, x = {model.apply(params, (1.0,))}')
