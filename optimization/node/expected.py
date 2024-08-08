import math as fm

import jax
import numpy as np
import scipy
from jax import numpy as jnp

from experiments.optimization.node import jet


def expected_value_jacfwd(func, loc, scale=1.0, n_iters=1):
    d = func
    m = 1.0
    res = d(loc) * m

    fact = 1
    for i in range(1, n_iters):
        print(i)
        fact *= i
        m = scipy.stats.norm.moment(i, loc=0.0, scale=scale)
        d = jax.jacfwd(d)
        if i % 2 == 0:
            dloc = d(loc)
            print(jnp.linalg.norm(dloc))
            res += dloc * m / fact
            #print(res)

    return res


def expected_value_jet(func, loc, scale=1.0, n_iters=1):
    moments = [scipy.stats.norm.moment(i, loc=0.0, scale=scale) for i in range(0, n_iters)]
    f0, cfs = jet.jet(func, (loc,), ((1.0,) + (n_iters-1) * (0.0,),))
    factorials = scipy.special.factorial(range(0, n_iters))
    res = f0
    print(f'{f0}, {cfs}')
    for i in range(1, n_iters):
        p = cfs[i-1] * moments[i] / factorials[i]
        print(f'p = {p}, d = {cfs[i-1]}, moment = {moments[i]}, factorial = {factorials[i]}')
        res += p
    return res


def expected_value_quad(func, loc, scale=1.0, n_iters=50, sigmas_n=3, abstol=1e-4, reltol=1e-3):
    pdf = lambda x: 1 / (fm.sqrt(2*fm.pi) * scale) * fm.exp(-(x - loc)**2 / (2*scale**2))
    f_pdf = lambda x: np.array(func(x)) * pdf(x)
    n = 3

    def adaptive_simpson(f, a, b):
        x_grid2 = np.linspace(a, b, 2*n-1)
        y_grid2 = [f(x) for x in x_grid2]

        x_grid = x_grid2[0::2]
        y_grid = y_grid2[0::2]
        i1 = scipy.integrate.simpson(y_grid, x=x_grid, axis=0)

        print(f'[{a}, {b}]')
        i2 = scipy.integrate.simpson(y_grid2, x=x_grid2, axis=0)
        if np.linalg.norm(i1 - i2) < abstol or np.linalg.norm(i1 - i2) / np.linalg.norm(i1) < reltol:
            return i2
        else:
            m = (a + b) / 2
            return adaptive_simpson(f, a, m) + adaptive_simpson(f, m, b)

    return adaptive_simpson(f_pdf, loc-sigmas_n*scale, loc+sigmas_n*scale)
