import math as fm

import jax
import lineax
from jax import numpy as jnp

from propagators._utils import pade_propagator_coefs


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


class AbstractWaveSpeedModel:

    def __call__(self, z):
        pass


class LinearWaveSpeedModel(AbstractWaveSpeedModel):

    def __init__(self, c0: float, slope_degrees: float):
        self.c0 = c0
        self.slope_degrees = slope_degrees

    def __call__(self, z):
        return self.c0 + z * jnp.tan(jnp.radians(self.slope_degrees))

    def _tree_flatten(self):
        dynamic = (self.slope_degrees,)
        static = {'c0': self.c0}
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        return cls(slope_degrees=dynamic[0], **static)


class RationalHelmholtzPropagator:

    def __init__(self, order: tuple[float, float], beta: float, dx_m: float, dz_m: float, x_n: int, z_n: int,
                 x_grid_scale: int, z_grid_scale: int, freq_hz: float, wave_speed: AbstractWaveSpeedModel, coefs=None):
        self.order = order
        self.beta = beta
        self.dx_m = dx_m
        self.dz_m = dz_m
        self.x_n = x_n
        self.z_n = z_n
        self.x_grid_scale = x_grid_scale
        self.z_grid_scale = z_grid_scale
        if coefs is not None:
            self.coefs_t = coefs
            self.coefs = jnp.array(coefs, dtype=complex)
        else:
            t = pade_propagator_coefs(pade_order=self.order, beta=self.beta, dx=self.dx_m)[0]
            a = [list(v) for v in t]
            self.coefs = jnp.array(a, dtype=complex)
            self.coefs_t = a
        self.freq_hz = freq_hz
        self.wave_speed = wave_speed
        self.het = jnp.array((2*jnp.pi*self.freq_hz/self.wave_speed(self.z_computational_grid()))**2 / self.beta ** 2 - 1.0, dtype=complex)

    def x_computational_grid(self):
        return jnp.arange(0, self.x_n) * self.dx_m

    def x_output_grid(self):
        return self.x_computational_grid()[::self.x_grid_scale]

    def z_computational_grid(self):
        return jnp.arange(0, self.z_n) * self.dz_m

    def z_output_grid(self):
        return self.z_computational_grid()[::self.z_grid_scale]

    def _tree_flatten(self):
        dynamic = (self.wave_speed,)
        static = {
            'order': self.order,
            'beta': self.beta,
            'dx_m': self.dx_m,
            'dz_m': self.dz_m,
            'x_n': self.x_n,
            'z_n': self.z_n,
            'coefs': self.coefs_t,
            'x_grid_scale': self.x_grid_scale,
            'z_grid_scale': self.z_grid_scale,
            'freq_hz': self.freq_hz
        }
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        unf = cls(wave_speed=dynamic[0], **static)
        return unf

    @jax.jit
    def _Crank_Nikolson_propagate_no_rho_4th_order(self, a, b, initial, lower_bound=(1, 0, 0), upper_bound=(0, 1, 0)):
        alpha = 1/12
        c_a = alpha * (self.beta * self.dz_m) ** 2 + a + alpha * a * (self.beta * self.dz_m) ** 2 * self.het
        c_b = alpha * (self.beta * self.dz_m) ** 2 + b + alpha * b * (self.beta * self.dz_m) ** 2 * self.het
        d_a = (self.beta * self.dz_m) ** 2 * (1 - 2 * alpha) - 2 * a + (a * (self.beta * self.dz_m) ** 2 - 2 * a * alpha * (self.beta * self.dz_m) ** 2) * self.het
        d_b = (self.beta * self.dz_m) ** 2 * (1 - 2 * alpha) - 2 * b + (b * (self.beta * self.dz_m) ** 2 - 2 * b * alpha * (self.beta * self.dz_m) ** 2) * self.het

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
        return jax.lax.fori_loop(0, len(self.coefs),
                          lambda i, val: self._Crank_Nikolson_propagate_no_rho_4th_order(
                              self.coefs[i][0], self.coefs[i][1], val), initial)

    @jax.jit
    def compute(self, initial):
        self.het = jnp.array((2*jnp.pi*self.freq_hz/self.wave_speed(self.z_computational_grid()))**2 / self.beta ** 2 - 1.0, dtype=complex)
        results = jnp.empty(shape=(round(self.x_n / self.x_grid_scale), round(self.z_n / self.z_grid_scale)), dtype=complex)
        results = results.at[0, :].set(initial[::self.z_grid_scale])

        def body_fun(i, val):
            y0, res = val
            y1 = self._step(y0)
            res = res.at[jnp.ceil(i / self.x_grid_scale).astype(int), :].set(y1[::self.z_grid_scale])
            return y1, res

        _, results = jax.lax.fori_loop(0, self.x_n, body_fun, (initial, results))

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
tree_util.register_pytree_node(LinearWaveSpeedModel,
                               LinearWaveSpeedModel._tree_flatten,
                               LinearWaveSpeedModel._tree_unflatten)