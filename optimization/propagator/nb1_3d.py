import jax
import jax.numpy as jnp
from jax import random
from jax import tree_util
from matplotlib.colors import Normalize
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from propagators._utils import pade_propagator_coefs

jax.config.update("jax_enable_x64", True)


class RationalApproximation:

    @classmethod
    def create_pade(cls, dx_beta: float, order=(7, 8)):
        pade_coefs_zipped, _ = pade_propagator_coefs(pade_order=order, dx=1, beta=dx_beta)
        a_coefs = [t[0] for t in pade_coefs_zipped[0:order[0]]]
        b_coefs = [t[1] for t in pade_coefs_zipped[0:order[1]]]
        return cls(order=order,
                   a_y_coefs=jnp.array(a_coefs),
                   a_z_coefs=jnp.array(a_coefs),
                   b_y_coefs=jnp.array(b_coefs),
                   b_z_coefs=jnp.array(b_coefs)
                   )

    @classmethod
    def create_random(cls, bounds=(-1, 1), order=(7, 8)):
        key = random.PRNGKey(1703)
        a_coefs = random.uniform(key, shape=(order[0],), minval=bounds[0], maxval=bounds[1])
        b_coefs = random.uniform(key, shape=(order[1],), minval=bounds[0], maxval=bounds[1])
        return cls(order=order, a_y_coefs=a_coefs, b_coefs=b_coefs)

    def __init__(self, order=(7, 8), a_y_coefs=None, a_z_coefs=None, b_y_coefs=None, b_z_coefs=None):
        self.order = order
        self.a_y_coefs = a_y_coefs if a_y_coefs is not None else jnp.zeros(order[0], dtype=complex)
        self.a_z_coefs = a_z_coefs if a_z_coefs is not None else jnp.zeros(order[0], dtype=complex)
        self.b_y_coefs = b_y_coefs if b_y_coefs is not None else jnp.zeros(order[0], dtype=complex)
        self.b_z_coefs = b_z_coefs if b_z_coefs is not None else jnp.zeros(order[0], dtype=complex)

    def _apply(self, y, z):
        return (jnp.prod(jnp.array([(1 + self.a_y_coefs[ind]*y)*(1 + self.a_z_coefs[ind]*z) for ind in range(self.order[0])]), axis=0) /
                jnp.prod(jnp.array([(1 + self.b_y_coefs[ind]*y)*(1 + self.b_z_coefs[ind]*z) for ind in range(self.order[1])]), axis=0))
    @jax.jit
    def __call__(self, y, z):
        return self._apply(y, z)

    def flat_coefs(self):
        return (list(self.a_y_coefs.real) + list(self.a_y_coefs.imag) +
                list(self.a_z_coefs.real) + list(self.a_z_coefs.imag) +
                list(self.b_y_coefs.real) + list(self.b_y_coefs.imag) +
                list(self.b_z_coefs.real) + list(self.b_z_coefs.imag)
                )

    def set_flat_coefs(self, flat_coefs):
        self.a_y_coefs = jnp.array([flat_coefs[i] + 1j * flat_coefs[i+self.order[0]] for i in range(0, self.order[0])])
        self.a_z_coefs = jnp.array([flat_coefs[i] + 1j * flat_coefs[i + self.order[0]] for i in range(2*self.order[0], 3*self.order[0])])
        self.b_y_coefs = jnp.array([flat_coefs[i] + 1j * flat_coefs[i + self.order[1]] for i in
                                    range(4 * self.order[0], 4 * self.order[0] + self.order[1])])
        self.b_z_coefs = jnp.array([flat_coefs[i] + 1j * flat_coefs[i + self.order[1]] for i in
                                    range(4 * self.order[0] + 2 * self.order[1], 4 * self.order[0] + 3 * self.order[1])])

    def _tree_flatten(self):
        dynamic = (self.a_y_coefs, self.a_z_coefs, self.b_y_coefs, self.b_z_coefs)
        static = {
            'order': self.order,
        }
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        unf = cls(a_y_coefs=dynamic[0], a_z_coefs=dynamic[1], b_y_coefs=dynamic[2], b_z_coefs=dynamic[3], **static)
        return unf


tree_util.register_pytree_node(RationalApproximation,
                               RationalApproximation._tree_flatten,
                               RationalApproximation._tree_unflatten)


def operator_symbol(dx_beta: float, xi_y: jax.Array, xi_z: jax.Array):
    return jnp.exp(1j*dx_beta*(jnp.sqrt(1 + xi_y + xi_z) - 1))


def loss(model: RationalApproximation, dx_beta, a, b, n=10):
    xi_y_grid = jnp.linspace(a, b, n)
    xi_z_grid = jnp.linspace(a, b, n)
    xi_y_2d_grid, xi_z_2d_grid = jnp.meshgrid(xi_y_grid, xi_z_grid)
    return jnp.linalg.norm(model(xi_y_2d_grid, xi_z_2d_grid) - operator_symbol(dx_beta, xi_y_2d_grid, xi_z_2d_grid))


def scipy_optimizer():
    model = RationalApproximation.create_pade(dx_beta=1.0, order=(7, 8))

    @jax.jit
    def scipy_loss(flatten_coefs):
        model.set_flat_coefs(flatten_coefs)
        return loss(model, dx_beta=1.0, a=0, b=3.0, n=100)

    @jax.jit
    def jac_loss(flatten_coefs):
        return jax.grad(scipy_loss)(flatten_coefs)

    m = minimize(
        method='L-BFGS-B',
        fun=scipy_loss,
        #args=(),
        x0=model.flat_coefs(),
        jac=jac_loss,
        callback=lambda intermediate_result: print(intermediate_result)
    )

    model = RationalApproximation.create_pade(dx_beta=1.0, order=(7, 8))
    model.set_flat_coefs(m.x)
    return model, m

res_model, m = scipy_optimizer()
print(m)


def plot_error(model, dx_beta, a, b):
    xi_y_grid = jnp.linspace(a, b, 1000)
    xi_z_grid = jnp.linspace(a, b, 1000)
    xi_y_2d_grid, xi_z_2d_grid = jnp.meshgrid(xi_y_grid, xi_z_grid)
    pade_model = RationalApproximation.create_pade(dx_beta=dx_beta, order=model.order)
    pointwise_error = abs(model(xi_y_2d_grid, xi_z_2d_grid) - operator_symbol(dx_beta, xi_y_2d_grid, xi_z_2d_grid))
    pointwise_error_pade = abs(pade_model(xi_y_2d_grid, xi_z_2d_grid) - operator_symbol(dx_beta, xi_y_2d_grid, xi_z_2d_grid))
    plt.figure()
    extent = [xi_y_grid[0], xi_y_grid[-1], xi_z_grid[0], xi_z_grid[-1]]
    norm = Normalize(-4, 0)
    plt.imshow(jnp.log10(pointwise_error), norm=norm, extent=extent, cmap=plt.get_cmap('jet'))
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.imshow(jnp.log10(pointwise_error_pade), norm=norm, extent=extent, cmap=plt.get_cmap('jet'))
    plt.grid(True)
    plt.show()

    # plt.figure()
    # plt.plot(xi_grid, jnp.log(model(xi_grid)).real, xi_grid, jnp.log(pade_model(xi_grid)).real)
    # plt.grid(True)
    # plt.show()
    #
    # plt.figure()
    # plt.scatter([a.real for a in pade_model.a_coefs], [a.imag for a in pade_model.a_coefs])
    # plt.scatter([b.real for b in pade_model.b_coefs], [b.imag for b in pade_model.b_coefs])
    # plt.xlim([-1.0, 1.0])
    # plt.ylim([-0.1, 0.1])
    #
    # plt.grid(True)
    # plt.show()
    #
    # plt.figure()
    # plt.scatter([a.real for a in model.a_coefs], [a.imag for a in model.a_coefs])
    # plt.scatter([b.real for b in model.b_coefs], [b.imag for b in model.b_coefs])
    # plt.xlim([-1.0, 1.0])
    # plt.ylim([-0.1, 0.1])
    # plt.grid(True)
    # plt.show()


plot_error(res_model, 1.0, 0, 10)
