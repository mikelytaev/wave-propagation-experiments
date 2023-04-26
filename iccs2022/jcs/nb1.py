import numpy as np

import pyximport
from pymoo.algorithms.soo.nonconvex.es import ES

pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)

from pymoo.optimize import minimize
from pymoo.termination import get_termination
from problems import UnconditionalOptimization


problem = UnconditionalOptimization(
    order=(6, 7),
    theta_max_degrees=22,
    dx_wl=50,
    dz_wl=0.25
)

# algorithm_de = DE(
#     pop_size=20,
#     sampling=LHS(),
#     variant="DE/best/1/exp",
#     CR=0.9,
#     dither="vector",
#     jitter=True,
# )

# algorithm = PSO(pop_size=100)

#ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

#algorithm = RVEA(ref_dirs)

algorithm = ES(n_offsprings=50, rule=1.0 / 7.0)

termination = get_termination("n_gen", 100000000)

res = minimize(problem,
               algorithm,
               seed=1,
               termination=termination,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))