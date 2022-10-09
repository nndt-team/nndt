import jax
import jax.numpy as jnp
import numpy as onp
from anytree import NodeMixin
from colorama import Fore
from jax.random import PRNGKeyArray

from nndt.math_core import grid_in_cube2, uniform_in_cube, take_each_n
from nndt.space2 import BBoxNode, node_method
from nndt.space2 import FileSource, Object3D
from nndt.space2.transformation import AbstractTransformation

import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk


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

    @node_method("save_mesh(filepath, {name, array})")
    def save_mesh(self, filepath: str, name_value: dict):
        surface = self.mesh._loader.mesh

        for keys, values in name_value.items():
            if isinstance(values, (onp.ndarray, onp.generic, jnp.ndarray, jnp.generic)):
                if values.ndim == 1:
                    data_ = numpy_to_vtk(num_array=values, deep=True, array_type=vtk.VTK_FLOAT)
                    data_.SetName(keys)
                    surface.GetPointData().AddArray(data_)
                else:
                    raise NotImplementedError
            elif values is list:
                data_ = numpy_to_vtk(num_array=values, deep=True, array_type=vtk.VTK_FLOAT)
                data_.SetName(keys)
                surface.GetPointData().AddArray(data_)
            else:
                raise NotImplementedError

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filepath)
        writer.SetInputData(surface)
        writer.Update()
        writer.Write()

    @node_method("sampling_eachN(count=N, step=1, shift=0) -> (ns_index[N], ns_xyz[N])")
    def sampling_eachN(self, count=1, step=1, shift=0) -> (jnp.ndarray, jnp.ndarray):
        index_set, array = take_each_n(self.mesh._loader.points,
                                       count=count, step=step, shift=shift)
        ret_array = self.transform.xyz_ps2ns(array)

        return index_set, ret_array

class SDTNode(MethodSetNode):
    def __init__(self, object_3d: Object3D,
                 sdt: FileSource,
                 transform: AbstractTransformation, parent: BBoxNode = None):
        super(SDTNode, self).__init__('sdt', parent=parent)
        self.object_3d = object_3d
        assert (sdt.loader_type == 'sdt')
        self.sdt = sdt
        self.transform = transform

    @node_method("xyz2sdt(ns_xyz[...,3]) -> ns_sdt[...,1]")
    def xyz2sdt(self, ns_xyz: jnp.ndarray) -> jnp.ndarray:
        ps_xyz = self.transform.xyz_ns2ps(ns_xyz)
        ps_sdt = self.sdt._loader.request(ps_xyz)
        ns_sdt = self.transform.sdt_ps2ns(ps_sdt)
        return ns_sdt