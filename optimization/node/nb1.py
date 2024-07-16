from scipy.optimize import minimize, differential_evolution
import math as fm
import jax.numpy as jnp
import numpy as np


freq = 50
f = lambda z: -jnp.array(1500 + 0.1*z)
max_res = minimize(f, x0=[0.0], bounds=[(0, 1000)])
print(max_res.x[0])

result_ga = differential_evolution(
    func=f,
    bounds=[(0, 1000)],
    popsize=30,
    disp=True,
    recombination=1,
    strategy='randtobest1exp',
    tol=1e-5,
    maxiter=10000,
    polish=False
)
print(result_ga)
