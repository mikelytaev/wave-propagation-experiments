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


class RationalHelmholtzPropagator:

    def __init__(self, order: tuple[float, float], k0: float, dx_m: float, dz_m: float, z_n: int, x_max_m: float, refractive_index, coefs=None):
        self.order = order
        self.k0 = k0
        self.dx_m = dx_m
        self.dz_m = dz_m
        self.refractive_index = refractive_index
        self.z_n = z_n
        self.het = self.refractive_index(jnp.arange(0, self.z_n) * self.dz_m) - 1.0
        self.x_max_m = x_max_m
        if coefs is not None:
            self.coefs_t = coefs
            self.coefs = jnp.array(coefs, dtype=complex)
        else:
            t = pade_propagator_coefs(pade_order=self.order, beta=self.k0, dx=self.dx_m)[0]
            a = [list(v) for v in t]
            self.coefs = jnp.array(a, dtype=complex)
            self.coefs_t = a

    def _tree_flatten(self):
        dynamic = (self.refractive_index,)
        static = {
            'order': self.order,
            'k0': self.k0,
            'dx_m': self.dx_m,
            'dz_m': self.dz_m,
            'z_n': self.z_n,
            'coefs': self.coefs_t,
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
