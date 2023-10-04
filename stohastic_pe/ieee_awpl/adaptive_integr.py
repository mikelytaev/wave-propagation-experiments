import scipy.integrate
import math as fm
import numpy as np


sigma = 4
mu = 10
def func(x):
    print(f"{len(x)} {sum(range(0, len(x)+1))}")
    return np.sin(x/2) * 1/(sigma*fm.sqrt(2*fm.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))


print(scipy.integrate.quadrature(func, mu-3*sigma, mu+3*sigma, vec_func=True, tol=1e-03, rtol=1e-03))
