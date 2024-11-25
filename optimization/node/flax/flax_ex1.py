from typing import Sequence
import jax
from jax import random, numpy as jnp
from flax import linen as nn
import matplotlib.pyplot as plt


def func(x):
    x *= 100
    return 100*jnp.sin(x*0.1)
    #return x*jnp.sin(2*x) + jnp.cos(x)


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

key1, key2 = random.split(random.key(0), 2)

model = ExplicitMLP(features=[40]*3 + [1])
params = model.init(key2, jnp.ones((1, 1)))


x_learn_grid = jnp.linspace(0, 1, 200).reshape((200, 1))


def operator(f_grid):
    return jnp.diff(f_grid, axis=0) + f_grid[0:-1]**3


def loss(params):
    m_v = model.apply(params, x_learn_grid)
    f_v = func(x_learn_grid)
    return jnp.linalg.norm((m_v) - (f_v))


import optax
learning_rate = 0.01
tx = optax.adam(learning_rate=learning_rate)

opt_state = tx.init(params)
loss_grad_fn = jax.value_and_grad(loss)

for i in range(3001):
  loss_val, grads = loss_grad_fn(params)
  updates, opt_state = tx.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  if i % 10 == 0:
    print('Loss step {}: '.format(i), loss_val)


x_test_grid = jnp.linspace(0, 1, 1000).reshape((1000, 1))
plt.figure(figsize=(6, 3.2))
plt.plot(x_test_grid, model.apply(params, x_test_grid))
plt.plot(x_test_grid, func(x_test_grid))
plt.show()
plt.grid(True)