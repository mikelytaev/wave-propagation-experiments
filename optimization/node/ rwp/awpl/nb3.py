import jax
import matplotlib.pyplot as plt

from utils import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt
#logging.basicConfig(level=logging.DEBUG)


def evaporation_operator(height_m: float, model: RWPModel):
    return model.apply_N_profile(EvaporationDuctModel(height_m=height_m))


def evaporation_loss0(height_m: float, model: RWPModel, measure):
    val = evaporation_operator(height_m, model)
    return Bartlett_loss(val, measure)


inv_model = RWPModel(params=RWPComputationalParams(
        max_range_m=150000,
        max_height_m=250,
        dx_m=100,
        dz_m=1
    ))
measure = evaporation_operator(10.0, inv_model)
measure = add_noise(measure, 30)

inv_model.apply_N_profile(EvaporationDuctModel(height_m=11))

landscape = [evaporation_loss0(h, inv_model, measure) for h in jnp.linspace(5, 30, 100)]
plt.plot(landscape)
plt.show()

@jax.jit
def hessp_loss(x, v):
    return hvp(evaporation_loss0, (x,), (v,))

m = minimize(
    method='L-BFGS-B',
    fun=evaporation_loss0,
    args=(inv_model, measure),
    x0=[25],
    jac=jax.grad(evaporation_loss0),
    #hessp=hessp_loss
)
print(m)