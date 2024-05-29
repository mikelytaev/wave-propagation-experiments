from typing import Callable

import diffrax
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Complex128
import jax
import jax.random as jr
import optax
from matplotlib.colors import Normalize
import cmath as cm


import matplotlib.pyplot as plt



class DiscreteFunction1D(eqx.Module):
    x_left: float = eqx.field(static=True)
    x_right: float = eqx.field(static=True)
    vals: Complex128[Array, "n"]

    @classmethod
    def discretise_fn(cls, x_left: float, x_right: float, n: int, fn: Callable):
        if n < 2:
            raise ValueError("Must discretise [x0, x_final] into at least two points")
        vals = jax.vmap(fn)(jnp.linspace(x_left, x_right, n))
        return cls(x_left, x_right, vals)

    @classmethod
    def discretise_fn2(cls, x_left: float, x_right: float, n: int, fn: Callable):
        if n < 2:
            raise ValueError("Must discretise [x0, x_final] into at least two points")
        vals = fn(jnp.linspace(x_left, x_right, n))
        return cls(x_left, x_right, vals)

    @property
    def dx(self):
        return (self.x_right - self.x_left) / (self.vals.shape[-1] - 1)

    def binop(self, other, fn):
        if isinstance(other, DiscreteFunction1D):
            if self.x_left != other.x_left or self.x_right != other.x_right:
                raise ValueError("Mismatched spatial discretisations")
            other = other.vals
        return DiscreteFunction1D(self.x_left, self.x_right, fn(self.vals, other))

    def __add__(self, other):
        return self.binop(other, lambda x, y: x + y)

    def __mul__(self, other):
        return self.binop(other, lambda x, y: x * y)

    def __radd__(self, other):
        return self.binop(other, lambda x, y: y + x)

    def __rmul__(self, other):
        return self.binop(other, lambda x, y: y * x)

    def __sub__(self, other):
        return self.binop(other, lambda x, y: x - y)

    def __rsub__(self, other):
        return self.binop(other, lambda x, y: y - x)


def munk_profile_jax(z_grid_m, ref_sound_speed: float = 1500, ref_depth: float = 1300, eps_: float = 0.00737):
    z_ = 2 * (z_grid_m - ref_depth) / ref_depth
    return ref_sound_speed * (1 + eps_ * (z_ - 1 + jnp.exp(-z_)))


class ProfileModel(eqx.Module):
    t: jax.Array

    def __init__(self, ref_sound_speed: float = 1500, ref_depth: float = 1300):
        self.t = jnp.array([ref_sound_speed, ref_depth])

    def __call__(self):
        het = DiscreteFunction1D.discretise_fn(0, 2000, 2000,
                                               lambda z: (1500/munk_profile_jax(z, self.t[0], self.t[1]))**2-1).vals
        return het


model = ProfileModel(
    ref_sound_speed=1500,
    ref_depth=1300
)

ys = model()

@eqx.filter_value_and_grad
def grad_loss_vertical(model: ProfileModel, z_batch_perm, yi):
    y_pred = model()
    m = yi - y_pred[z_batch_perm]
    return (jnp.mean(m.real ** 2) + jnp.mean(m.imag ** 2)) / jnp.linalg.norm(yi)**2


def dataloader(yi, batch_size: int, *, key):
    indices = jnp.arange(yi.shape[0])
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        end = batch_size
        batch_perm = perm[0:end]
        yield batch_perm, yi[batch_perm]


key = jr.PRNGKey(12345)
batch_size = 32

optim = optax.adabelief(20.0)
model = ProfileModel(
    ref_sound_speed=1290.0,
    ref_depth=1407.0
)
opt_state = optim.init(model)
print(opt_state)

@eqx.filter_jit
def make_step(model, z_batch_perm, f_batch, opt_state):
    loss, grads = grad_loss_vertical(model, z_batch_perm, f_batch)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

loader = dataloader(ys, batch_size, key=key)
for step in range(500):
    z_batch_perm, f_batch = next(loader)
    loss, model, opt_state = make_step(model, z_batch_perm, f_batch, opt_state)
    print(f'Loss = {loss}, ref_sound_speed = {model.t[0]}, ref_depth = {model.t[1]}')
