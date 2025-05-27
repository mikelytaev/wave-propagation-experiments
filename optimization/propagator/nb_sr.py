import numpy as np
from pysr import PySRRegressor


omega = 2 * np.pi * 1000
c = 1500
n_points = 100
dx_beta = 1.0

xi_range = np.linspace(0, 0.1, n_points)
X = xi_range.reshape(xi_range.shape[0], 1)

y = np.exp(1j*dx_beta * np.sqrt(
    1 - X**2
))

model = PySRRegressor(
    niterations=100,
    binary_operators=['+', '*'],
    unary_operators=["inv(x) = 1/x"],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    populations=20,
    verbosity=1
)

model.fit(X, y)

print("Best equations found:")
print(model.equations_)
