import time

import lineax
import numpy as np
import jax.numpy as jnp
import equinox as eqx
import jax
import jax.random as jr
import optax
from matplotlib.colors import Normalize
import cmath as cm


import matplotlib.pyplot as plt

from propagators._utils import pade_propagator_coefs
import math as fm

from uwa.environment import munk_profile
from uwa.source import GaussSource


class AbstractRefractiveIndexModel:

    def __call__(self, *args, **kwargs):
        pass


class EmptyRefractiveIndexModel(AbstractRefractiveIndexModel):

    def __call__(self, z_grid_m):
        return z_grid_m*0.0j + 1.0

    def _tree_flatten(self):
        dynamic = ()
        static = {}
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        unf = cls()
        return unf


class MunkProfileModel(AbstractRefractiveIndexModel):

    def __init__(self, ref_sound_speed: float = 1500, ref_depth: float = 1300):
        self.ref_sound_speed = ref_sound_speed
        self.ref_depth = ref_depth

    @jax.jit
    def __call__(self, z_grid_m: jnp.ndarray):
        eps_: float = 0.00737
        z_ = 2 * (z_grid_m - self.ref_depth) / self.ref_depth
        sound_speed = self.ref_sound_speed * (1 + eps_ * (z_ - 1 + jnp.exp(-z_)))
        return (1500/sound_speed)**2

    def _tree_flatten(self):
        dynamic = (self.ref_sound_speed, self.ref_depth)
        static = {}
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        unf = cls(ref_sound_speed=dynamic[0], ref_depth=dynamic[1])
        return unf


class RationalHelmholtzPropagator:

    def __init__(self, order: tuple[float, float], k0: float, dx_m: float, dz_m: float, refractive_index: AbstractRefractiveIndexModel, z_n: int, x_max_m: float, coefs=None):
        self.order = order
        self.k0 = k0
        self.dx_m = dx_m
        self.dz_m = dz_m
        self.refractive_index = refractive_index
        self.z_n = z_n
        self.het = self.refractive_index(jnp.arange(0, self.z_n) * self.dz_m) - 1.0
        self.x_max_m = x_max_m
        if coefs is not None:
            self.coefs = coefs
        else:
            t = pade_propagator_coefs(pade_order=self.order, k0=self.k0, dx=self.dx_m)[0]
            a = [list(v) for v in t]
            self.coefs = jnp.array(a, dtype=complex)

    def _tree_flatten(self):
        dynamic = (self.refractive_index,)
        static = {
            'order': self.order,
            'k0': self.k0,
            'dx_m': self.dx_m,
            'dz_m': self.dz_m,
            'z_n': self.z_n,
            'coefs': self.coefs,
            #'refractive_index': self.refractive_index,
            'x_max_m': self.x_max_m
        }
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        unf = cls(refractive_index=dynamic[0], **static)
        return unf

    @jax.jit
    def _Crank_Nikolson_propagate_no_rho_4th_order(self, a, b, initial, lower_bound=(1, 0, 0), upper_bound=(0, 1, 0)):
        alpha = 1/12
        c_a = alpha * (self.k0 * self.dz_m) ** 2 + a + alpha * a * (self.k0 * self.dz_m) ** 2 * self.het
        c_b = alpha * (self.k0 * self.dz_m) ** 2 + b + alpha * b * (self.k0 * self.dz_m) ** 2 * self.het
        d_a = (self.k0 * self.dz_m) ** 2 * (1 - 2 * alpha) - 2 * a + (a * (self.k0 * self.dz_m) ** 2 - 2 * a * alpha * (self.k0 * self.dz_m) ** 2) * self.het
        d_b = (self.k0 * self.dz_m) ** 2 * (1 - 2 * alpha) - 2 * b + (b * (self.k0 * self.dz_m) ** 2 - 2 * b * alpha * (self.k0 * self.dz_m) ** 2) * self.het

        rhs = d_a * initial
        rhs = rhs.at[1::].set(rhs[1::] + c_a[:-1:] * initial[:-1:])
        rhs = rhs.at[:-1:].set(rhs[:-1:] + c_a[1::] * initial[1::])
        d_b = d_b.at[0].set(lower_bound[0])
        d_b = d_b.at[-1].set(upper_bound[1])
        diag_1 = c_b[1::]
        diag_1 = diag_1.at[0].set(lower_bound[1])
        diag_m1 = c_b[:-1:]
        diag_m1 = diag_m1.at[-1].set(upper_bound[0])
        rhs = rhs.at[0].set(lower_bound[2])
        rhs = rhs.at[-1].set(upper_bound[2])
        tridiag_op = lineax.TridiagonalLinearOperator(d_b, diag_m1, diag_1)
        res = lineax.linear_solve(tridiag_op, rhs)
        return res.value

    @jax.jit
    def _step(self, initial):
        initial = jax.lax.fori_loop(0, len(self.coefs),
                          lambda i, val: self._Crank_Nikolson_propagate_no_rho_4th_order(
                              self.coefs[i][0], self.coefs[i][1], val), initial)
        return initial

    @jax.jit
    def compute(self, initial):
        count = int(fm.ceil(self.x_max_m / self.dx_m))
        self.het = self.refractive_index(jnp.arange(0, self.z_n) * self.dz_m) - 1.0
        results = jnp.empty(shape=(count, len(initial)), dtype=complex)

        def body_fun(i, val):
            y0, res = val
            y1 = self._step(y0)
            res = res.at[i, :].set(y1)
            return y1, res

        _, results = jax.lax.fori_loop(0, count, body_fun, (initial, results))

        return results


from jax import tree_util
tree_util.register_pytree_node(MunkProfileModel,
                               MunkProfileModel._tree_flatten,
                               MunkProfileModel._tree_unflatten)
tree_util.register_pytree_node(EmptyRefractiveIndexModel,
                               EmptyRefractiveIndexModel._tree_flatten,
                               EmptyRefractiveIndexModel._tree_unflatten)
tree_util.register_pytree_node(RationalHelmholtzPropagator,
                               RationalHelmholtzPropagator._tree_flatten,
                               RationalHelmholtzPropagator._tree_unflatten)


src = GaussSource(
    freq_hz=50,
    depth_m=500,
    beam_width_deg=10,
    elevation_angle_deg=0
)
k0 = 2*jnp.pi*src.freq_hz/1500
z_grid = np.linspace(0, 4000, 1000)

het = munk_profile(z_grid)
het = (1500/het)**2-1

model = RationalHelmholtzPropagator(
    k0=k0,
    dx_m=100,
    dz_m=z_grid[1]-z_grid[0],
    z_n=1000,
    refractive_index=MunkProfileModel(),
    order=(7, 8),
    x_max_m=10000
)
model2 = RationalHelmholtzPropagator(
    k0=k0,
    dx_m=100,
    dz_m=z_grid[1]-z_grid[0],
    z_n=1000,
    refractive_index=MunkProfileModel(),
    order=(7, 8),
    x_max_m=10000
)
init = src.aperture(k0=k0, z=z_grid)

start = time.time()
r = model.compute(init)
end = time.time()
print(f'time = {end-start}')

start = time.time()
#model.refractive_index = MunkProfileModel()
r = model.compute(init)
end = time.time()
print(f'time = {end-start}')

start = time.time()
model.refractive_index.ref_depth = 1
r = model.compute(init)
end = time.time()
print(f'time = {end-start}')

plt.figure(figsize=(6, 3.2))
plt.imshow(
    20*jnp.log10(jnp.abs(r+1e-16)).T,
    norm=Normalize(vmin=-120, vmax=-40),
    aspect='auto',
    cmap=plt.get_cmap('jet')
)
plt.colorbar()
plt.show()


def loss_vertical(model: RationalHelmholtzPropagator, z_batch_perm, yi, init):
    t = model.compute(init)
    y_pred = t[-1, :]
    m = jnp.log10(jnp.abs((yi+1e-16)/(y_pred[z_batch_perm]+1e-16)))
    return jnp.mean(m.real ** 2)


def dataloader(yi, batch_size: int, *, key):
    indices = jnp.arange(int(yi.shape[0]/3)) + int(yi.shape[0]/3)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        end = batch_size
        batch_perm = perm[0:end]
        yield batch_perm, yi[batch_perm]


key = jr.PRNGKey(12345)
batch_size = 32

optim = optax.adabelief(20.0)
training_model = RationalHelmholtzPropagator(
    k0=k0,
    dx_m=100,
    dz_m=z_grid[1]-z_grid[0],
    z_n=1000,
    refractive_index=MunkProfileModel(),
    order=(7, 8),
    x_max_m=10000
)
model2.compute(init)
opt_state = optim.init(training_model)
print(opt_state)


#@jax.jit
def make_step(model, z_batch_perm, f_batch, opt_state, init):
    loss = loss_vertical(model, z_batch_perm, f_batch, init)
    grad = jax.grad(loss_vertical)(model, z_batch_perm, f_batch, init)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


loader = dataloader(r[-1, :], batch_size, key=key)
for step in range(500):
    z_batch_perm, f_batch = next(loader)
    start = time.time()
    loss, training_model, opt_state = make_step(training_model, z_batch_perm, f_batch, opt_state, init)
    end = time.time()
    print(f'Loss = {loss}, ref_sound_speed = {model.refractive_index.ref_sound_speed}, ref_depth = {model.refractive_index.ref_depth}, time = {end-start}')
