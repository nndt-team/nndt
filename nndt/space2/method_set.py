from typing import Optional

import jax
import jax.numpy as jnp
import numpy as onp
import vtk
from colorama import Fore
from jax.random import PRNGKeyArray
from vtkmodules.util.numpy_support import numpy_to_vtk

from nndt.math_core import grid_in_cube2, uniform_in_cube, take_each_n
from nndt.space2 import FileSource
from nndt.space2 import node_method
from nndt.space2.abstracts import AbstractTreeElement, AbstractBBoxNode
from nndt.space2.transformation import AbstractTransformation


def _get_class_hierarchy(obj):
    class_hierarchy = [obj.__class__]
    while len(class_hierarchy[-1].__bases__) > 0:
        class_hierarchy = class_hierarchy + [class_hierarchy[-1].__bases__[0]]
    return class_hierarchy


class MethodNode(AbstractTreeElement):
    def __init__(self, name: str, docstring: Optional[str], parent=None):
        super(MethodNode, self).__init__(name, _print_color=Fore.RESET, _nodetype='M', parent=parent)
        self.docstring = docstring if docstring is not None else name

    def __repr__(self):
        return self._print_color + f'{self.docstring}' + Fore.RESET


class MethodSetNode(AbstractTreeElement):
    def __init__(self, name: str, parent=None, ):
        super(MethodSetNode, self).__init__(name, _print_color=Fore.YELLOW, _nodetype='MS', parent=parent)

    def _post_attach(self, parent):
        if parent is not None:
            setattr(parent, self.name, self)

    def _post_detach(self, parent):
        if parent is not None:
            if hasattr(parent, self.name):
                delattr(parent, self.name)


class SamplingNode(MethodSetNode):

    def __init__(self, parent: AbstractBBoxNode = None):
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
    def __init__(self, object_3d: AbstractBBoxNode,
                 mesh: FileSource,
                 transform: AbstractTransformation, parent: AbstractBBoxNode = None):
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
    def __init__(self, object_3d: AbstractBBoxNode,
                 sdt: FileSource,
                 transform: AbstractTransformation, parent: AbstractBBoxNode = None):
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
