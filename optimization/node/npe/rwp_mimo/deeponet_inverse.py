import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from typing import Callable, Tuple
import optax
from jax import random
from matplotlib import pyplot as plt

jax.config.update("jax_enable_x64", True)


class DeepONet(nnx.Module):

    def __init__(self, rngs: nnx.Rngs, samples_num: int = 128, interact_size=128, branch_scale=(0, 1), trunk_scale=(0, 1),
                 branch_hidden_dims: Tuple[int] = (128, 128, 128), trunk_hidden_dims: Tuple[int] = (128, 128, 128)):
        self.branch_net = BranchNet(samples_num=samples_num, interact_size=interact_size, rngs=rngs, scale=branch_scale, hidden_dims=branch_hidden_dims)
        self.trunk_net = TrunkNet(interact_size=interact_size, rngs=rngs, scale=trunk_scale, hidden_dims=trunk_hidden_dims)
        self.bias = nnx.Param(jnp.zeros((1,)))
    
    def __call__(self, v: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        branch_output = self.branch_net(v)
        trunk_output = self.trunk_net(x)
        return jnp.sum(branch_output * trunk_output, axis=-1) + self.bias

class BranchNet(nnx.Module):

    def __init__(self, rngs: nnx.Rngs, samples_num: int = 128, interact_size=128,
                 hidden_dims: Tuple[int] = (128, 128, 128), scale=(0, 1)):
        self.scale = scale
        self.layers = []
        dims = samples_num, *hidden_dims, interact_size
        for ind in range(len(dims) - 1):
            s_in = dims[ind]
            s_out = dims[ind + 1]
            self.layers.append(nnx.Linear(s_in, s_out, rngs=rngs))


    def __call__(self, v: jnp.ndarray) -> jnp.ndarray:
        x = (v - self.scale[0]) / self.scale[1]
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nnx.relu(x)
        x = self.layers[-1](x)
        return x

class TrunkNet(nnx.Module):

    def __init__(self, rngs: nnx.Rngs, interact_size=128, hidden_dims: Tuple[int] = (128, 128, 128), scale=(0, 1)):
        self.scale = scale
        self.layers = []
        dims = 1, *hidden_dims, interact_size
        for ind in range(len(dims)-1):
            s_in = dims[ind]
            s_out = dims[ind+1]
            self.layers.append(nnx.Linear(s_in, s_out, rngs=rngs))

    def __call__(self, v: jnp.ndarray) -> jnp.ndarray:
        v = (v - self.scale[0]) / self.scale[1]
        x = v.reshape(-1, 1)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nnx.relu(x)
        x = self.layers[-1](x)
        return x


def learn_inverse_G(
        G: Callable,
        generator: Callable,
        model: nnx.Module,
        grid: jnp.ndarray,
        max_epoch_num=1000,
        batch_size=100,
        tx: optax.GradientTransformation=None
) -> Callable:
    key = random.PRNGKey(1703)
    gen = generator(key)
    batched_generator = jax.vmap(generator)
    batched_G = jax.vmap(G)
    args_grid = grid

    if tx is None:
        tx = optax.adam(learning_rate=0.01, b1=0.9)
    optimizer = nnx.Optimizer(model, tx)
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average('loss'),
    )

    @nnx.jit
    def cycle_loss_fn(inv_G: DeepONet, batched_args, args_grid):
        batched_vals = batched_G(batched_args)
        predictions = jax.vmap(inv_G, in_axes=(0, None))(batched_vals, args_grid)
        return jnp.mean(jnp.square(predictions - batched_args))
        #return jnp.mean(jnp.square(batched_G(predictions) - batched_vals))

    @nnx.jit
    def train_step(model: DeepONet, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batched_args, args_grid):
        grad_fn = nnx.value_and_grad(cycle_loss_fn)
        loss, grads = grad_fn(model, batched_args, args_grid)
        #metrics.update(loss=loss, logits=logits)
        optimizer.update(grads)
        return loss

    losses = []
    for epoch in range(max_epoch_num):
        key = random.split(key, 1)[0]
        keys = random.split(key, batch_size)
        batched_args = batched_generator(keys)

        l = train_step(model, optimizer, metrics, batched_args, args_grid)
        print(f"epoch={epoch}, loss={l}")
        losses.append(l)


    return model, losses


if __name__ == "__main__":

    def G(n: jnp.ndarray) -> jnp.ndarray:
        #return n**3 + 2*n**2 + 3*n
        return jnp.diff(n**3)

    grid = jnp.linspace(0, 1, 128)

    def generator(key=random.PRNGKey(17031993)) -> jnp.ndarray:
        return grid * jax.random.uniform(key, minval=0.01, maxval=2)

    model = DeepONet(rngs=nnx.Rngs(1703), samples_num=G(generator()).shape[0], interact_size=300)
    grid = jnp.linspace(0, 1, generator().shape[0])
    G_inv, losses = learn_inverse_G(G, generator, model, grid)

    plt.plot(np.log10(losses))
    plt.show()

    key = random.PRNGKey(17031993)
    u = generator(key)
    v = G(u)
    grid2 = jnp.linspace(0, 1, 100)
    uu = G_inv(v, grid2)
    #vv = G(uu)

    plt.plot(grid, u)
    plt.plot(grid2, uu)
    plt.show()

    # plt.plot(v)
    # plt.plot(vv)
    # plt.show()