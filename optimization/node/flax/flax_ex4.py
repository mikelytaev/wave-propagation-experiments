from copy import deepcopy

from experiments.optimization.node.flax.utils import MLPWaveSpeedModel, ExplicitMLP
import jax.numpy as jnp
import jax
import jax.random as random

import matplotlib.pyplot as plt


#m = MLPWaveSpeedModel()
mlp = ExplicitMLP(features=[40]*3 + [1])
#params = mlp.params
key1, key2 = random.split(random.key(0), 2)
x = random.uniform(key1, (3,1))
params = mlp.init(key2, x)

def true_ssp(z):
    return 100*jnp.sin(z*0.1)


z_grid_m = jnp.linspace(0, 100, 200).reshape(200,1)

def loss0(params):
    return jnp.linalg.norm(jnp.ravel(true_ssp(z_grid_m)) - jnp.ravel(mlp.apply(params, z_grid_m/100)))


import optax
learning_rate = 0.01
tx = optax.adam(learning_rate=learning_rate)

opt_state = tx.init(params)
loss_grad_fn = jax.value_and_grad(loss0)

for i in range(1000):
  loss_val, grads = loss_grad_fn(params)
  updates, opt_state = tx.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  if i % 10 == 0:
    print('Loss step {}: '.format(i), loss_val)


plt.figure(figsize=(6, 3.2))
plt.plot(mlp.apply(params, z_grid_m/100), z_grid_m)
plt.plot(true_ssp(z_grid_m), z_grid_m)
plt.show()
plt.grid(True)
