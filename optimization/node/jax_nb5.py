from typing import List

import jax
from jax import tree_util


class A:

    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def _tree_flatten(self):
        dynamic = (self.a,)
        static = {
            'b': self.b
        }
        return dynamic, static

    @jax.jit
    def mul(self):
        return self.a * self.b

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        return cls(a=dynamic[0], **static)


tree_util.register_pytree_node(A,
                               A._tree_flatten,
                               A._tree_unflatten)


class B:

    def __init__(self, c: float, d_dyn: A, d_static: A, l: List = []):
        self.c = c
        self.d_dyn = d_dyn
        self.d_static = d_static
        self.l = l

    def _tree_flatten(self):
        dynamic = (self.c, self.d_dyn, self.l)
        static = {
            "d_static": self.d_static
        }
        return dynamic, static

    @jax.jit
    def mul(self):
        return self.c * self.d_dyn.mul() * self.d_static.mul() * len(self.l)

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        return cls(c=dynamic[0], d_dyn=dynamic[1], l=dynamic[2], **static)


tree_util.register_pytree_node(B,
                               B._tree_flatten,
                               B._tree_unflatten)


a1 = A(5.0, 10.0)
a2 = A(2.0, 5.0)
b1 = B(2.0, a1, a2)
print(b1.mul())
b1.l = [0.0]
print(b1.mul())
