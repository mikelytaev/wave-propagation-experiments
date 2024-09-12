from typing import Sequence
import jax
from jax import random, numpy as jnp
from flax import linen as nn
import matplotlib.pyplot as plt


def func(x):
    return x*jnp.sin(2*x) + jnp.cos(x)


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
x = random.uniform(key1, (3,1))

model = ExplicitMLP(features=[30]*5 + [1])
params = model.init(key2, x)
y = model.apply(params, x)


x_learn_grid = jnp.linspace(-5, 5, 200).reshape((200, 1))


def operator(f_grid):
    return jnp.diff(f_grid, axis=0) + f_grid[0:-1]**3


def loss(params):
    m_v = model.apply(params, x_learn_grid)
    f_v = func(x_learn_grid)
    return jnp.linalg.norm(operator(m_v) - operator(f_v))


import optax
learning_rate = 0.001
tx = optax.adam(learning_rate=learning_rate)

opt_state = tx.init(params)
loss_grad_fn = jax.value_and_grad(loss)

for i in range(3001):
  loss_val, grads = loss_grad_fn(params)
  updates, opt_state = tx.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  if i % 10 == 0:
    print('Loss step {}: '.format(i), loss_val)


x_test_grid = jnp.linspace(-7, 7, 1000).reshape((1000, 1))
plt.figure(figsize=(6, 3.2))
plt.plot(x_test_grid, model.apply(params, x_test_grid))
plt.plot(x_test_grid, func(x_test_grid))
plt.show()
plt.grid(True)