import jax
import jax.numpy as jnp
from anytree import NodeMixin
from colorama import Fore
from jax.random import PRNGKeyArray

from nndt.math_core import grid_in_cube2, uniform_in_cube
from nndt.space2 import BBoxNode, node_method
from nndt.space2 import FileSource, Object3D
from nndt.space2.transformation import AbstractTransformation


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

        from nndt.space2 import initialize_method_node
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


class MeshNode(MethodSetNode):
    def __init__(self, object_3d: Object3D,
                 mesh: FileSource,
                 transform: AbstractTransformation, parent: BBoxNode = None):
        super(MeshNode, self).__init__('mesh', parent=parent)
        self.object_3d = object_3d
        assert (mesh.loader_type == 'mesh_obj')
        self.mesh = mesh
        self.transform = transform

    @node_method("index2xyz(ns_index[...,1]) -> ns_xyz[...,3]")
    def index2xyz(self, ns_index: jnp.ndarray) -> jnp.ndarray:
        result_ps = jnp.take(self.mesh._loader.points, ns_index)
        result_ns = self.transform.xyz_ps2ns(result_ps)
        return result_ns

    @node_method("xyz2index(ns_index[...,3]) -> ns_xyz[...,1]")
    def xyz2index(self, ns_index: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError("Processing with KDTree is not implemented yet. Sorry.")
