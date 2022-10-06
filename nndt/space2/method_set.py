import jax
import jax.numpy as jnp
from anytree import NodeMixin
from colorama import Fore
from jax.random import PRNGKeyArray

from nndt.math_core import grid_in_cube2, uniform_in_cube
from nndt.space2 import BBoxNode, node_method


class MethodSetNode(NodeMixin):
    def __init__(self, name: str, parent=None,
                 _print_color: str = Fore.RESET,
                 _nodetype: str = 'MS'):
        super(MethodSetNode, self).__init__()

        self.name = name
        self.parent = parent
        self._print_color = _print_color
        self._nodetype = _nodetype

    def __repr__(self):
        return self._print_color + f'{self._nodetype}:{self.name}' + Fore.RESET

    def _post_attach(self, parent):
        if parent is not None:
            setattr(parent, self.name, self)

        from space2 import initialize_method_node
        initialize_method_node(self)

    def _post_detach(self, parent):
        if parent is not None:
            if hasattr(parent, self.name):
                delattr(parent, self.name)


class SamplingNode(MethodSetNode):

    def __init__(self, parent: BBoxNode = None):
        super(SamplingNode, self).__init__('sampling', parent=parent)

    @node_method("grid(spacing=(D,H,W)) -> xyz[D,H,W,3]")
    def grid(self, spacing: (int, int, int) = (2, 2, 2)) -> jnp.ndarray:
        lower, upper = self.parent.bbox
        basic_cube = grid_in_cube2(spacing=spacing,
                                   lower=jnp.array(lower),
                                   upper=jnp.array(upper))
        return basic_cube

    @node_method("grid_with_shackle(key, N) -> xyz[N,3]")
    def grid_with_shackle(self, rng_key: PRNGKeyArray, spacing: (int, int, int) = (2, 2, 2), sigma=0.1) -> jnp.ndarray:
        assert (sigma > 0.00000001)
        lower, upper = self.parent.bbox
        shift_xyz = jax.random.normal(rng_key, shape=(3,)) * sigma
        basic_cube = grid_in_cube2(spacing=spacing,
                                   lower=jnp.array(lower),
                                   upper=jnp.array(upper)) + shift_xyz
        return basic_cube

    @node_method("uniform(key, N) -> xyz[N,3]")
    def uniform(self, rng_key: PRNGKeyArray, count: int = 100) -> jnp.ndarray:
        lower, upper = self.parent.bbox
        basic_cube = uniform_in_cube(rng_key, count=count, lower=lower, upper=upper)
        return basic_cube
