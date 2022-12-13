from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as onp
import vtk
from colorama import Fore
from jax.random import KeyArray
from vtkmodules.util.numpy_support import numpy_to_vtk

from nndt.math_core import grid_in_cube, grid_in_cube2, take_each_n, uniform_in_cube
from nndt.space2.abstracts import AbstractBBoxNode, AbstractTreeElement, node_method
from nndt.space2.filesource import FileSource
from nndt.space2.implicit_representation import ImpRepr
from nndt.space2.transformation import AbstractTransformation
from nndt.space2.utils import calc_ret_shape
from nndt.vizualize import ANSIConverter


def _get_class_hierarchy(obj):
    class_hierarchy = [obj.__class__]
    while len(class_hierarchy[-1].__bases__) > 0:
        class_hierarchy = class_hierarchy + [class_hierarchy[-1].__bases__[0]]
    return class_hierarchy


class MethodNode(AbstractTreeElement):
    def __init__(self, name: str, docstring: Optional[str], parent=None):
        super(MethodNode, self).__init__(
            name, _print_color=Fore.RESET, _nodetype="M", parent=parent
        )
        self.docstring = docstring if docstring is not None else name

    def __repr__(self):
        return self._print_color + f"{self.docstring}" + Fore.RESET

    def _repr_html_(self):
        return (
            '<p>'
            + f'<span style=\"color:{ANSIConverter(self._print_color, type="Fore").to_rgb()}\">'
            + f'{self.docstring}'
            + '</span>'
            + '</p>'
        )


class MethodSetNode(AbstractTreeElement):
    def __init__(
        self,
        name: str,
        parent=None,
    ):
        super(MethodSetNode, self).__init__(
            name, _print_color=Fore.YELLOW, _nodetype="MS", parent=parent
        )

    def _post_attach(self, parent):
        if parent is not None:
            setattr(parent, self.name, self)

    def _post_detach(self, parent):
        if parent is not None:
            if hasattr(parent, self.name):
                delattr(parent, self.name)


class SamplingMethodSetNode(MethodSetNode):
    def __init__(self, parent: AbstractBBoxNode = None):
        super(SamplingMethodSetNode, self).__init__("sampling", parent=parent)

    @node_method("sampling_grid(spacing=(D,H,W)) -> ns_xyz[D,H,W,3]")
    def sampling_grid(self, spacing: (int, int, int) = (2, 2, 2)) -> jnp.ndarray:
        """
        Sample points from a bounding box (bbox) according to the vertex of the uniform mesh.
        This method work for the normalized space.

        Data transformation
        sampling_grid(spacing=(D,H,W)) -> ns_xyz[D,H,W,3]

        :param spacing: Number of slices for each coordinate of the bbox.
        :return: 4-dimensional tensor, the last axis is (x,y,z) point position.
        """
        lower, upper = self.parent.bbox
        basic_cube = grid_in_cube2(
            spacing=spacing, lower=jnp.array(lower), upper=jnp.array(upper)
        )
        return basic_cube

    @node_method("sampling_grid_with_noise(key, spacing=(D,H,W), sigma) -> ns_xyz[N,3]")
    def sampling_grid_with_noise(
        self, rng_key: KeyArray, spacing: (int, int, int) = (2, 2, 2), sigma=0.1
    ) -> jnp.ndarray:
        """
        Sample points from a bounding box (bbox) according to the vertex of the uniform mesh.
        Then, the method adds Gaussian noise N(0, sigma) to the point positions.
        This method work for the normalized space.

        Data transformation
        sampling_grid_with_noise(key, spacing=(D,H,W), sigma) -> ns_xyz[N,3]

        :param rng_key: a key for JAX's random generators
        :param spacing: Number of slices for each coordinate of the bbox
        :param sigma: Standard deviation of the Gaussian spatial noise
        :return: 4-dimensional tensor, the last axis is (x,y,z) point position
        """
        lower, upper = self.parent.bbox
        shift_xyz = jax.random.normal(rng_key, shape=(3,)) * sigma
        basic_cube = (
            grid_in_cube2(
                spacing=spacing, lower=jnp.array(lower), upper=jnp.array(upper)
            )
            + shift_xyz
        )
        return basic_cube

    @node_method("sampling_uniform(key, N) -> ns_xyz[N,3]")
    def sampling_uniform(self, rng_key: KeyArray, count: int = 100) -> jnp.ndarray:
        """
        Sample points from a bounding box (bbox) according to the multidimensional uniform distribution.

        Data transformation
        sampling_uniform(key, N) -> ns_xyz[N,3]

        :param rng_key: a key for JAX's random generators
        :param count: number of points for generation
        :return: 2-dimensional tensor, the last axis is (x,y,z) point position

        """
        lower, upper = self.parent.bbox
        basic_cube = uniform_in_cube(rng_key, count=count, lower=lower, upper=upper)
        return basic_cube


class MeshObjMethodSetNode(MethodSetNode):
    def __init__(
        self,
        object_3d: AbstractBBoxNode,
        mesh: FileSource,
        transform: AbstractTransformation,
        parent: AbstractBBoxNode = None,
    ):
        super(MeshObjMethodSetNode, self).__init__("mesh", parent=parent)
        self.object_3d = object_3d
        assert mesh.loader_type == "mesh_obj"
        self.mesh = mesh
        self.transform = transform

    @node_method("surface_xyz() -> xyz[N,3]")
    def surface_xyz(self) -> jnp.ndarray:
        """
        Return the position of the surface mesh vertexes.

        Data transformation
        surface_xyz() -> xyz[N,3]

        :return: tensor, the last axis is (x,y,z) point position
        """
        ps_xyz = self.mesh._loader.points
        ns_xyz = self.transform.transform_xyz_ps2ns(ps_xyz)
        return ns_xyz

    @node_method("surface_ind2xyz(ind[..,1]) -> ns_xyz[..,3]")
    def surface_ind2xyz(self, ind: jnp.ndarray) -> jnp.ndarray:
        """
        Convert indexes of the surface mesh vertexes to their coordinates.

        This transformation keeps the shape of the tensor.
        Transformation is performed along the last axis.

        Data transformation
        surface_ind2xyz(ind[..,1]) -> ns_xyz[..,3]

        :param ind: indexes of points
        :return: tensor, the last axis is (x,y,z) point position
        """
        assert ind.shape[-1] == 1 or ind.ndim == 1
        ret_shape = calc_ret_shape(ind, 3)
        ind_flat = ind.flatten()

        result_ps = jnp.take(self.mesh._loader.points, ind_flat, axis=0)
        result_ns = self.transform.transform_xyz_ps2ns(result_ps)

        result_ns = result_ns.reshape(ret_shape)
        return result_ns

    @node_method("surface_xyz2ind(ns_xyz[..,3]) -> ns_dist[..,1], ns_ind[..,1]")
    def surface_xyz2ind(self, ns_xyz: jnp.ndarray) -> (jnp.ndarray, jnp.ndarray):
        """
        Convert coordinates of the surface mesh vertexes to their indexes.
        If coordinates not corresponding to a mesh vertex, then the nearest vertex is detected.
        This transformation keeps the shape of the tensor. Transformation is performed along the last axis.
        Note, this method work wraps low-level KDTree implementation.

        Data transformation
        surface_xyz2ind(ns_xyz[..,3]) -> ns_dist[..,1], ns_ind[..,1]

        :param ns_xyz: points in the normalized space
        :return: distances and indexes of the surface mesh points

        """
        assert ns_xyz.shape[-1] == 3
        ret_shape = calc_ret_shape(ns_xyz, 1)
        ns_xyz_flat = ns_xyz.reshape((-1, 3))

        ps_xyz = self.transform.transform_xyz_ns2ps(ns_xyz_flat)
        ps_dist, ind = self.mesh._loader.kdtree.query(onp.array(ps_xyz))
        ind = jnp.array(ind).reshape(ret_shape)
        ns_dist = self.transform.transform_sdt_ps2ns(ps_dist)
        ns_dist = jnp.array(ns_dist).reshape(ret_shape)

        return ns_dist, ind

    @node_method("save_mesh(filepath, {name, array})")
    def save_mesh(self, filepath: str, name_value: dict):
        """
        Save a surface mesh to .vtp file.
        Dictionary may include data for storage. The dictionary key is an array name, the dictionary value is an array for storage.

        Data transformation
        save_mesh(filepath, {name, array})

        :param filepath: Path to the .vtp file
        :param name_value: Dictionary with name of vtk-arrays and data for the storage.
        :return:
        """
        surface = self.mesh._loader.mesh

        for keys, values in name_value.items():
            if isinstance(values, (onp.ndarray, onp.generic, jnp.ndarray, jnp.generic)):
                if values.ndim == 1:
                    data_ = numpy_to_vtk(
                        num_array=values, deep=True, array_type=vtk.VTK_FLOAT
                    )
                    data_.SetName(keys)
                    surface.GetPointData().AddArray(data_)
                else:
                    raise NotImplementedError
            elif values is list:
                data_ = numpy_to_vtk(
                    num_array=values, deep=True, array_type=vtk.VTK_FLOAT
                )
                data_.SetName(keys)
                surface.GetPointData().AddArray(data_)
            else:
                raise NotImplementedError

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filepath)
        writer.SetInputData(surface)
        writer.Update()
        writer.Write()

    @node_method(
        "sampling_eachN_from_mesh(count=N, step=M, shift=K) -> (ns_ind[N], ns_xyz[N])"
    )
    def sampling_eachN_from_mesh(
        self, count=1, step=1, shift=0
    ) -> (jnp.ndarray, jnp.ndarray):
        """
        Sample points from the mesh. This is a deterministic sampler, that returns each `step` point from the mesh vertex sequence.
        If an iteration pointer overcomes the array length, then it brings it to the beginning.

        Data transformation
        sampling_eachN_from_mesh(count=N, step=M, shift=K) -> (ns_ind[N], ns_xyz[N])

        :param count: Number of the requested points
        :param step: Step of the iterator
        :param shift: Shift of the first iteration from the zero indexes.
        :return: array of indexes and array of values

        """
        index_set, array = take_each_n(
            self.mesh._loader.points, count=count, step=step, shift=shift
        )
        ret_array = self.transform.transform_xyz_ps2ns(array)

        return index_set, ret_array


class SDTMethodSetNode(MethodSetNode):
    def __init__(
        self,
        object_3d: AbstractBBoxNode,
        sdt: Union[FileSource, ImpRepr],
        transform: AbstractTransformation,
        parent: AbstractBBoxNode = None,
    ):
        super(SDTMethodSetNode, self).__init__("sdt", parent=parent)
        self.object_3d = object_3d
        assert isinstance(sdt, ImpRepr) or sdt.loader_type == "sdt"
        self.sdt = sdt
        self.transform = transform

    @node_method("surface_xyz2sdt(ns_xyz[..,3]) -> ns_sdt[..,1]")
    def surface_xyz2sdt(self, ns_xyz: jnp.ndarray) -> jnp.ndarray:
        """
        Converts coordinates of points to signed distance from the points to an object surface.
        The result of the method is a tensor with values of the signed distance function (SDT).
        This transformation keeps the shape of the tensor.
        Transformation is performed along the last axis.

        Data transformation
        surface_xyz2sdt(ns_xyz[..,3]) -> ns_sdt[..,1]

        :param ns_xyz: coordinates in the normalized space
        :return: tensor with distances from points to the surface
        """
        ps_xyz = self.transform.transform_xyz_ns2ps(ns_xyz)
        ps_sdt = self.sdt._loader.request(ps_xyz)
        ns_sdt = self.transform.transform_sdt_ps2ns(ps_sdt)
        return ns_sdt

    @node_method(
        "surface_xyz2localsdt(ns_xyz[3], spacing=(D,H,W), scale=1.) -> ns_xyz[D,H,W,3], ns_localsdt[D,H,W,1]"
    )
    def surface_xyz2localsdt(
        self, ns_xyz: jnp.ndarray, spacing=(2, 2, 2), scale=1.0
    ) -> (jnp.ndarray, jnp.ndarray):
        """
        This method is an encoding of geometrical features for the point's surroundings.
        The method creates boxes around the requested points.
        Then, it defines a uniform grid (UG) with the requested `spacing` inside the bbox.
        Each vertex of the uniform grid (UG) is converted to the value of the signed distance function.
        Thus, the method convert points to signed distance tensors (SDF) of their surroundings.

        Data transformation:
        surface_xyz2localsdt(ns_xyz[3], spacing=(D,H,W), scale=1.) -> ns_xyz[D,H,W,3], ns_localsdt[D,H,W,1]

        :param ns_xyz: coordinates of points in normalized space
        :param spacing: Number of slices for each coordinates of the box
        :param scale: scale factor for the box in normalized space
        :return: tensor of UG and SDF
        """
        ns_cube = grid_in_cube(
            spacing=spacing, scale=scale, center_shift=(0.0, 0.0, 0.0)
        )
        ns_cube = ns_cube + ns_xyz
        ns_cube = ns_cube.reshape((-1, 3))
        ps_cube = self.transform.transform_xyz_ns2ps(ns_cube)
        ps_local_sdt = self.sdt._loader.request(ps_cube)
        ns_local_sdt = self.transform.transform_sdt_ps2ns(ps_local_sdt)
        ns_local_sdt = ns_local_sdt.reshape(spacing)[:, :, :, jnp.newaxis]
        ns_cube = ns_cube.reshape((spacing[0], spacing[1], spacing[2], 3))
        return ns_cube, ns_local_sdt


class ColorMethodSetNode(MethodSetNode):
    def __init__(
        self,
        object_3d: AbstractBBoxNode,
        mesh: FileSource,
        transform: AbstractTransformation,
        parent: AbstractBBoxNode = None,
    ):
        super(ColorMethodSetNode, self).__init__("mesh_colors", parent=parent)
        self.object_3d = object_3d
        assert mesh.loader_type == "mesh_obj"
        self.mesh = mesh
        self.transform = transform

    @node_method("surface_rgba() -> xyz[N,4]")
    def surface_rgba(self) -> jnp.ndarray:
        """
        Get colors of all surface vertexes

        Data transformation
        surface_rgba() -> xyz[N,4]

        :return: colors in RGBA format
        """
        rgba = self.mesh._loader.rgba
        return rgba

    @node_method("surface_ind2rgba(ind[..,1]) -> rgba[..,4]")
    def surface_ind2rgba(self, ind: jnp.ndarray) -> jnp.ndarray:
        """
        Convert indexes of the surface vertex to their colors

        Data transformation
        surface_ind2rgba(ind[..,1]) -> rgba[..,4]

        :param ind: index of surface vertexes
        :return: colors in RGBA format
        """
        assert ind.shape[-1] == 1 or ind.ndim == 1
        ret_shape = calc_ret_shape(ind, 4)

        ind_flat = ind.flatten()
        rgba = jnp.take(self.mesh._loader.rgba, ind_flat, axis=0)

        rgba = rgba.reshape(ret_shape)
        return rgba

    @node_method("surface_xyz2rgba(ns_xyz[..,3]) -> rgba[..,4]")
    def surface_xyz2rgba(self, ns_xyz: jnp.ndarray) -> jnp.ndarray:
        """
        Convert coordinates of points to colors of the nearest vertex on the surface.

        Data transformation
        surface_xyz2rgba(ns_xyz[..,3]) -> rgba[..,4]

        :param ns_xyz: coordinates of points in normalized space
        :return: colors in RGBA format
        """
        assert ns_xyz.shape[-1] == 3
        ret_shape = calc_ret_shape(ns_xyz, 4)

        ns_xyz_flat = ns_xyz.reshape((-1, 3))
        ps_xyz_flat = self.transform.transform_xyz_ns2ps(ns_xyz_flat)
        ps_dist, ind = self.mesh._loader.kdtree.query(onp.array(ps_xyz_flat))
        color = jnp.take(self.mesh._loader.rgba, ind, axis=0)

        color = color.reshape(ret_shape)
        return color
