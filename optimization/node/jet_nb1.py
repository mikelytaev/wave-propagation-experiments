import jax
import jax.numpy as np

from jax.experimental import jet


h0 = 0.5
f = lambda z: ((1.1 + 1j)*np.sin(z*np.cos(np.array([z, z**2+1j])))).real

df = f
for i in range(1, 8):
    print(df(h0))
    df = jax.jacfwd(df)


print('!!!!!!!!!!')
f0, (f1, f2, f3, f4, f5, f6) = jet.jet(f, (h0,), ((1.0, 0.0, 0.0, 0.0, 0.0, 0.0),))
print(f0,  f(h0))

print(f1)
print(f2)
print(f3)
print(f4)
print(f5)
print(f6)
