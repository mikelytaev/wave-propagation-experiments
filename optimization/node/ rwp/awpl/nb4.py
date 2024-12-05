import jax
import matplotlib.pyplot as plt

from experimental.rwp_jax import PiecewiseLinearNProfileModel
from utils import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt
#logging.basicConfig(level=logging.DEBUG)

jax.config.update("jax_enable_x64", True)


z_grid = jnp.linspace(0, 250, 21)
def pwl_operator(vals: jnp.ndarray, model: RWPModel):
    return model.apply_N_profile(PiecewiseLinearNProfileModel(z_grid, jnp.concat((jnp.array([10.0]), vals))))


def pwl_loss0(vals: jnp.ndarray, model: RWPModel, measure):
    val = pwl_operator(vals, model)
    return Bartlett_loss(val, measure)


def pwl_loss1(vals: jnp.ndarray, model: RWPModel, measure):
    return (jnp.linalg.norm(jnp.diff(jnp.concat((jnp.array([10.0]), vals)))))**2


def random_gen():
    key = jax.random.key(1703)
    while True:
        key, subkey = jax.random.split(key)
        yield key


gamma = 1.0e-3
def loss(vals: jnp.ndarray, model: RWPModel, measure):
    #seed = int(jnp.linalg.norm(vals**10))
    #print(f'{seed}')
    #measure = add_noise(measure, 30, jax.random.key(seed))
    return pwl_loss0(vals, model, measure) + gamma*pwl_loss1(vals, model, measure)


inv_model = RWPModel(params=ComputationalParams(
        max_range_m=20000,
        max_height_m=250,
        dx_m=1000,
        dz_m=1
    ),
    measure_points_x=[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    measure_points_z=[2, 10, 20, 30, 40, 60, 100, 125, 150, 170, 190, 200, 220, 240, 250],
)

true_profile = PiecewiseLinearNProfileModel(jnp.linspace(0.0, 250, 3), jnp.array([10, 0, 20]))
measure = inv_model.apply_N_profile(true_profile)
measure = add_noise(measure, 30)

# landscape = [evaporation_loss0(h, inv_model, measure) for h in jnp.linspace(5, 30, 100)]
# plt.plot(landscape)
# plt.show()

# @jax.jit
# def hessp_loss(x, v):
#     return hvp(evaporation_loss0, (x,), (v,))

x0 = true_profile(z_grid[1::]) + 5*2*(jax.random.uniform(jax.random.PRNGKey(0), shape=z_grid[1::].shape)-0.5) + 3
m = minimize(
    method='L-BFGS-B',
    fun=loss,
    args=(inv_model, measure),
    x0=x0,
    jac=jax.grad(loss),
    #hessp=hessp_loss,
    callback=lambda xk: print(f'{xk}, {pwl_loss0(xk, inv_model, measure)} {gamma*pwl_loss1(xk, inv_model, measure)}'),

)
print(m)

plt.plot(jnp.concat((jnp.array([10.0]), m.x)), z_grid)
plt.plot(true_profile(z_grid), z_grid)
plt.plot(jnp.concat((jnp.array([10.0]), x0)), z_grid)
plt.show()