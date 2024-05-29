from typing import Callable

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.lax as lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping
import time


jax.config.update("jax_enable_x64", True)


# Represents the interval [x0, x_final] discretised into n equally-spaced points.
class SpatialDiscretisation(eqx.Module):
    x0: float = eqx.field(static=True)
    x_final: float = eqx.field(static=True)
    vals: Float[Array, "n"]

    @classmethod
    def discretise_fn(cls, x0: float, x_final: float, n: int, fn: Callable):
        if n < 2:
            raise ValueError("Must discretise [x0, x_final] into at least two points")
        vals = jax.vmap(fn)(jnp.linspace(x0, x_final, n))
        return cls(x0, x_final, vals)

    @property
    def δx(self):
        return (self.x_final - self.x0) / (len(self.vals) - 1)

    def binop(self, other, fn):
        if isinstance(other, SpatialDiscretisation):
            if self.x0 != other.x0 or self.x_final != other.x_final:
                raise ValueError("Mismatched spatial discretisations")
            other = other.vals
        return SpatialDiscretisation(self.x0, self.x_final, fn(self.vals, other))

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


def laplacian(y: SpatialDiscretisation) -> SpatialDiscretisation:
    y_next = jnp.roll(y.vals, shift=1)
    y_prev = jnp.roll(y.vals, shift=-1)
    Δy = (y_next - 2 * y.vals + y_prev) / (y.δx**2)
    # Dirichlet boundary condition
    Δy = Δy.at[0].set(0)
    Δy = Δy.at[-1].set(0)
    return SpatialDiscretisation(y.x0, y.x_final, Δy)


# Problem
@jax.jit
def vector_field(t, y, args):
    return (1 - y) * laplacian(y)


#@jax.jit
def f():
    n = 50
    term = diffrax.ODETerm(vector_field)
    ic = lambda x: x**2

    # Spatial discretisation
    x0 = -1
    x_final = 1
    y0 = SpatialDiscretisation.discretise_fn(x0, x_final, n, ic)

    # Temporal discretisation
    t0 = 0
    t_final = 1
    δt = 0.0001
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t_final, 50))

    # Tolerances
    rtol = 1e-10
    atol = 1e-10
    stepsize_controller = diffrax.PIDController(
        pcoeff=0.3, icoeff=0.4, rtol=rtol, atol=atol, dtmax=0.001
    )

    solver = diffrax.Tsit5()
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0,
        t_final,
        δt,
        y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=None,
    )

    return sol.ys

start = time.time()
f()
end = time.time()
print(f"Computation time: {end - start}")

start = time.time()
f()
end = time.time()
print(f"Computation time: {end - start}")

start = time.time()
f(100)
end = time.time()
print(f"Computation time: {end - start}")
