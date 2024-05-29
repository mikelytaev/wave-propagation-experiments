import time
from typing import Callable

import diffrax
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, BacksolveAdjoint
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Complex128
import jax
import jax.random as jr
import optax
from matplotlib.colors import Normalize
import cmath as cm


import matplotlib.pyplot as plt


class GaussSource(eqx.Module):
    freq_hz: float = eqx.field(static=True)
    depth_m: float = eqx.field(static=True)
    beam_width_deg: float = eqx.field(static=True)
    elevation_angle_deg: float = eqx.field(static=True)

    def __init__(self, *, freq_hz, depth_m, beam_width_deg, elevation_angle_deg):
        self.freq_hz = freq_hz
        self.depth_m = depth_m
        self.beam_width_deg = beam_width_deg
        self.elevation_angle_deg = elevation_angle_deg

    def aperture(self, k0, z):
        elevation_angle_rad = self.elevation_angle_deg * cm.pi / 180
        ww = cm.sqrt(2 * cm.log(2)) / (k0 * cm.sin(self.beam_width_deg * cm.pi / 180 / 2))
        return (1 / (cm.sqrt(cm.pi) * ww) * jnp.exp(-1j * k0 * jnp.sin(elevation_angle_rad) * z) *
                jnp.exp(-((z - self.depth_m) / ww) ** 2))

    def max_angle_deg(self):
        return self.beam_width_deg + abs(self.elevation_angle_deg)


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


def laplacian(y: DiscreteFunction1D) -> DiscreteFunction1D:
    y_next = jnp.roll(y.vals, shift=1)
    y_prev = jnp.roll(y.vals, shift=-1)
    d2y = (y_next - 2 * y.vals + y_prev) / (y.dx**2)
    # Dirichlet boundary condition
    d2y = d2y.at[0].set(0)
    d2y = d2y.at[-1].set(0)
    return DiscreteFunction1D(y.x_left, y.x_right, d2y)


def munk_profile_jax(z_grid_m, ref_sound_speed: float = 1500, ref_depth: float = 1300, eps_: float = 0.00737):
    z_ = 2 * (z_grid_m - ref_depth) / ref_depth
    return ref_sound_speed * (1 + eps_ * (z_ - 1 + jnp.exp(-z_)))


class NarrowParabolicEq(eqx.Module):
    freq_hz: float = eqx.field(static=True)
    k0: float = eqx.field(static=True)
    c0: float = eqx.field(static=True)
    t: jax.Array
    # ref_sound_speed: float = eqx.field()
    # ref_depth: float = eqx.field()

    def __init__(self, freq_hz: float, ref_sound_speed: float = 1500, ref_depth: float = 1300):
        self.freq_hz = freq_hz
        self.c0 = 1500.0
        self.k0 = 2*jnp.pi*freq_hz/self.c0
        self.t = jnp.array([ref_sound_speed, ref_depth], dtype=float)

    def _rhs(self, t, y, args):
        het = DiscreteFunction1D.discretise_fn(0, y.x_right, len(y.vals),
                                               lambda z: (self.c0/munk_profile_jax(z, self.t[0], self.t[1]))**2 - 1).vals
        het = het.at[0].set(0)
        het = het.at[-1].set(0)
        res = 1j / (2*self.k0) * laplacian(y) + 1j*self.k0*DiscreteFunction1D(0.0, y.x_right, het)*y
        return res

    def __call__(self, ts):
        src = GaussSource(
            freq_hz=self.freq_hz,
            depth_m=100,
            beam_width_deg=3,
            elevation_angle_deg=0
        )
        y0 = DiscreteFunction1D.discretise_fn(0, 2000, 500, lambda z: src.aperture(self.k0, z))
        term = ODETerm(self._rhs)
        solver = Dopri5()
        x_left = ts[0]
        x_right = ts[-1]
        saveat = diffrax.SaveAt(ts=ts)
        solution = diffeqsolve(term, solver, t0=x_left, t1=x_right, dt0=0.1, y0=y0, saveat=saveat,
                               max_steps=100000000)
        return solution.ys


freq_hz = 50
model = NarrowParabolicEq(
    freq_hz=freq_hz
)

x_output_grid = jnp.linspace(0, 10000, 100)

start = time.time()
ys = model(x_output_grid)
end = time.time()
print(f'time = {end-start}')

start = time.time()
ys = model(x_output_grid)
end = time.time()
print(f'time = {end-start}')

start = time.time()
ys = model(x_output_grid)
end = time.time()
print(f'time = {end-start}')

plt.figure(figsize=(6, 3.2))
plt.imshow(
    20*jnp.log10(jnp.abs(ys.vals+1e-16)).T,
    norm=Normalize(vmin=-120, vmax=-40),
    aspect='auto',
    cmap=plt.get_cmap('jet')
)
plt.colorbar()
plt.show()


@eqx.filter_value_and_grad
def grad_loss_vertical(model: NarrowParabolicEq, ti: float, z_batch_perm, yi):
    y_pred = model(jnp.array([0.0, ti])).vals[-1,:]
    m = jnp.log10(jnp.abs((yi+1e-16)/(y_pred[z_batch_perm]+1e-16)))
    return jnp.mean(m.real ** 2)# + jnp.mean(m.imag ** 2)


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
model = NarrowParabolicEq(
    freq_hz=freq_hz,
    ref_sound_speed=1450.0,
    ref_depth=1335.0
)
opt_state = optim.init(model)
print(opt_state)

@eqx.filter_jit
def make_step(model, z_batch_perm, f_batch, x, opt_state):
    loss, grads = grad_loss_vertical(model, x, z_batch_perm, f_batch)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


loader = dataloader(ys.vals[-1,:], batch_size, key=key)
for step in range(500):
    z_batch_perm, f_batch = next(loader)
    start = time.time()
    loss, model, opt_state = make_step(model, z_batch_perm, f_batch, x_output_grid[-1], opt_state)
    end = time.time()
    print(f'Loss = {loss}, ref_sound_speed = {model.t[0]}, ref_depth = {model.t[1]}, time = {end-start}')
