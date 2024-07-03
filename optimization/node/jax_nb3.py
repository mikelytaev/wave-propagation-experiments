import jax
import jax.numpy as jnp
from jax.tree_util import tree_structure


class CustomClass:

    def __init__(self, x: jnp.ndarray, mul: bool):
        self.x = x
        self.mul = mul

    @jax.jit
    def calc(self, y):
        if self.mul:
          return self.x * y
        return y

    def _tree_flatten(self):
        children = (self.x,)  # arrays / dynamic values
        aux_data = {'mul': self.mul}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

from jax import tree_util
tree_util.register_pytree_node(CustomClass,
                               CustomClass._tree_flatten,
                               CustomClass._tree_unflatten)


init = jnp.array([1, 2, 3j])
c = CustomClass(init, True)
print(c.calc(jnp.array([1, 2, 3j])))


c.mul = False  # mutation is detected
print(c.calc(3))


cc = CustomClass(init, True)  # non-hashable x is supported
print(cc.calc(jnp.array([1, 2, 3j])))
