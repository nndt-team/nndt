import os.path
import os.path
import warnings

import jax
import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from math_core import grid_in_cube2, take_each_n, uniform_in_cube
from nndt.space.abstracts import *
from nndt.space.sources import MeshSource
from nndt.space.utils import downup_update_bbox
from nndt.space.vtk_wrappers import *


class MeshRepr(AbstractRegion, ExtendedNodeMixin, UnloadMixin):

    def __init__(self, parent: AbstractSource, surface_mesh2: SurfaceMesh,
                 physical_center: (float, float, float),
                 physical_bbox: ((float, float, float), (float, float, float)),
                 normed_center: (float, float, float),
                 normed_bbox: ((float, float, float), (float, float, float)),
                 scale_physical2normed: float,
                 _ndim=3,
                 _scale=1.,
                 name=""):
        super(MeshRepr, self).__init__(_ndim=_ndim,
                                       _bbox=normed_bbox,
                                       name=name)
        self.name = name
        self.parent = parent
        self._surface_mesh2 = surface_mesh2

        self.physical_center = physical_center
        self.physical_bbox = physical_bbox
        self.normed_center = normed_center
        self.normed_bbox = normed_bbox

        self.scale_physical2normed = scale_physical2normed

        self._print_color = Fore.GREEN

    @property
    def surface_mesh2(self):
        return self._surface_mesh2

    def unload_mesh(self):
        self._surface_mesh2.unload_data()

    def is_data_load(self):
        return self._surface_mesh2.is_data_load()

    def index_physical2normed(self, index: int) -> int:
        return index

    def index_normed2physical(self, index: int) -> int:
        return index

    def xyz_physical2normed(self, xyz: (float, float, float)) -> (float, float, float):
        X = (xyz[0] - self.physical_center[0]) / self.scale_physical2normed + self.normed_center[0]
        Y = (xyz[1] - self.physical_center[1]) / self.scale_physical2normed + self.normed_center[1]
        Z = (xyz[2] - self.physical_center[2]) / self.scale_physical2normed + self.normed_center[2]

        return (X, Y, Z)

    def xyz_normed2physical(self, xyz: (float, float, float)) -> (float, float, float):
        X = (xyz[0] - self.normed_center[0]) * self.scale_physical2normed + self.physical_center[0]
        Y = (xyz[1] - self.normed_center[1]) * self.scale_physical2normed + self.physical_center[1]
        Z = (xyz[2] - self.normed_center[2]) * self.scale_physical2normed + self.physical_center[2]

        return (X, Y, Z)

    @classmethod
    def load_mesh_and_bring_to_center(cls, source: MeshSource,
                                      padding_physical=(10, 10, 10),
                                      scale_physical2normed=50):
        surface_mesh2 = SurfaceMesh(source.filepath)
        Xmin, Xmax, Ymin, Ymax, Zmin, Zmax = surface_mesh2.mesh.GetBounds()
        surface_mesh2.unload_data()

        normed_center = (0., 0., 0.)
        physical_bbox = ((Xmin - padding_physical[0], Ymin - padding_physical[1], Zmin - padding_physical[2]),
                         (Xmax + padding_physical[0], Ymax + padding_physical[1], Zmax + padding_physical[2]))
        physical_center = (physical_bbox[0][0] + (physical_bbox[1][0] - physical_bbox[0][0]) / 2.,
                           physical_bbox[0][1] + (physical_bbox[1][1] - physical_bbox[0][1]) / 2.,
                           physical_bbox[0][2] + (physical_bbox[1][2] - physical_bbox[0][2]) / 2.)

        scale_physical2normed = scale_physical2normed

        normed_bbox = (((physical_bbox[0][0] - physical_center[0]) / scale_physical2normed + normed_center[0],
                        (physical_bbox[0][1] - physical_center[1]) / scale_physical2normed + normed_center[1],
                        (physical_bbox[0][2] - physical_center[2]) / scale_physical2normed + normed_center[2]),
                       ((physical_bbox[1][0] - physical_center[0]) / scale_physical2normed + normed_center[0],
                        (physical_bbox[1][1] - physical_center[1]) / scale_physical2normed + normed_center[1],
                        (physical_bbox[1][2] - physical_center[2]) / scale_physical2normed + normed_center[2]))

        repr = MeshRepr(source, surface_mesh2,
                        physical_center,
                        physical_bbox,
                        normed_center,
                        normed_bbox,
                        scale_physical2normed,
                        name="repr")

        downup_update_bbox(repr)

        return repr


class SaveMesh(AbstractMethod, ExtendedNodeMixin, UnloadMixin):

    def __init__(self, parent: MeshRepr):
        super(SaveMesh, self).__init__()
        self.name = "save_mesh"
        self.parent = parent

    def __repr__(self):
        return f'save_mesh(filepath, dict)'

    def __call__(self, filepath: str, name_value: dict):
        surface = self.parent.surface_mesh2.mesh

        for keys, values in name_value.items():
            if isinstance(values, (onp.ndarray, onp.generic)):
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


class Index2xyz(AbstractMethod, ExtendedNodeMixin, UnloadMixin):

    def __init__(self, parent: MeshRepr):
        super(Index2xyz, self).__init__()
        self.name = "index2xyz"
        self.parent = parent

        self._points = None

    def __repr__(self):
        return f'{self.name}(ns_index[1]) -> ns_xyz[3]'

    @property
    def points(self):
        if self._points is None:
            mesh = self.parent.surface_mesh2.mesh
            self._points = vtk_to_numpy(mesh.GetPoints().GetData())
        return self._points

    def unload_data(self):
        self._points = None

    def is_data_load(self):
        return (self._points is not None)

    def __call__(self, normed_index: int) -> (float, float, float):
        physical_index = self.parent.index_normed2physical(normed_index)
        physical_xyz = self.points[physical_index]
        normed_xyz = self.parent.xyz_physical2normed(physical_xyz)
        return normed_xyz


class PointColorRepr(ExtendedNodeMixin, UnloadMixin):

    def __init__(self, parent: MeshRepr):
        super(PointColorRepr, self).__init__()
        self.name = "point_color"
        self.parent = parent

        self._color_red = None
        self._color_green = None
        self._color_blue = None
        self._color_alpha = None

    def __repr__(self):
        return Fore.GREEN + f'{str(self.__class__.__name__)}("{self.name}")' + Fore.RESET

    @property
    def red(self):
        if self._color_red is None:
            self._load_all_data()
        return self._color_red

    @property
    def green(self):
        if self._color_green is None:
            self._load_all_data()
        return self._color_green

    @property
    def blue(self):
        if self._color_blue is None:
            self._load_all_data()
        return self._color_blue

    @property
    def alpha(self):
        if self._color_alpha is None:
            self._load_all_data()
        return self._color_alpha

    def unload_data(self):
        self._color_red = None
        self._color_green = None
        self._color_blue = None

    def is_data_load(self):
        return (self._color_red is not None) and \
               (self._color_green is not None) and \
               (self._color_blue is not None)

    def _load_all_data(self):

        filepath = self.parent.surface_mesh2.filepath
        num_of_points = self.parent.surface_mesh2.mesh.GetNumberOfPoints()

        self.file = Path(filepath)
        self.file.resolve(strict=True)
        _, file_extension = os.path.splitext(self.file)

        if file_extension == ".ply":
            r, g, b, a_ = PointColorRepr._load_colors_from_ply(filepath)
            if num_of_points == len(r):
                self._color_red = r
                self._color_green = g
                self._color_blue = b
                self._color_alpha = a_
            else:
                raise NotImplementedError()
        elif file_extension == ".obj":
            r, g, b, a_ = PointColorRepr._load_colors_from_obj(filepath)
            if num_of_points == len(r):
                self._color_red = r
                self._color_green = g
                self._color_blue = b
                self._color_alpha = a_
            else:
                raise NotImplementedError()

    @classmethod
    def _load_colors_from_obj(cls, filepath):
        red = []
        green = []
        blue = []
        alpha = []

        with open(filepath, 'r') as fl:
            for line in fl:
                if "v" in line:
                    tokens = line.split(" ")
                    if ("v" == tokens[0]) and (len(tokens) >= 7):
                        red.append(float(tokens[4].replace(',', '.')))
                        green.append(float(tokens[5].replace(',', '.')))
                        blue.append(float(tokens[6].replace(',', '.')))
                        alpha.append(1.)

        red = jnp.array(red)
        green = jnp.array(green)
        blue = jnp.array(blue)
        alpha = jnp.array(alpha)

        return red, green, blue, alpha

    @classmethod
    def _load_colors_from_ply(cls, filepath):
        red = []
        green = []
        blue = []
        alpha = []

        is_read_mode = False

        with open(filepath, 'r') as fl:
            for line in fl:
                if "end_header" in line:
                    is_read_mode = True
                if is_read_mode:
                    tokens = line.split(" ")
                    if len(tokens) >= 10:
                        red.append(float(tokens[6].replace(',', '.')))
                        green.append(float(tokens[7].replace(',', '.')))
                        blue.append(float(tokens[8].replace(',', '.')))
                        alpha.append(float(tokens[9].replace(',', '.')))

        red = jnp.array(red) / 255
        green = jnp.array(green) / 255
        blue = jnp.array(blue) / 255
        alpha = jnp.array(alpha) / 255

        return red, green, blue, alpha

    @classmethod
    def try_to_build_representation(cls, parent: MeshRepr):
        ret = None

        try:
            filepath = parent.surface_mesh2.filepath
            num_of_points = parent.surface_mesh2.mesh.GetNumberOfPoints()

            file = Path(filepath)
            file.resolve(strict=True)
            _, file_extension = os.path.splitext(file)

            filepath = parent.surface_mesh2.filepath
            if file_extension == ".ply":
                r, g, b, a = PointColorRepr._load_colors_from_obj(filepath)
                if not (len(r) == num_of_points): raise NotImplementedError()
                if not (len(g) == num_of_points): raise NotImplementedError()
                if not (len(b) == num_of_points): raise NotImplementedError()
                if not (len(a) == num_of_points): raise NotImplementedError()
            if file_extension == ".obj":
                r, g, b, a = PointColorRepr._load_colors_from_obj(filepath)
                if not (len(r) == num_of_points): raise NotImplementedError()
                if not (len(g) == num_of_points): raise NotImplementedError()
                if not (len(b) == num_of_points): raise NotImplementedError()
                if not (len(a) == num_of_points): raise NotImplementedError()

            ret = PointColorRepr(parent)

        except BaseException as err:
            warnings.warn("Color representation cannot be loaded:(")

        return ret


class SamplingGrid(AbstractMethod, ExtendedNodeMixin):

    def __init__(self, parent: AbstractRegion):
        super(SamplingGrid, self).__init__()
        self.name = "sampling_grid"
        self.parent = parent

    def __repr__(self):
        return f'sampling_grid(spacing=(D,H,W)) -> xyz[D,H,W,3]'

    def __call__(self, spacing: (int, int, int) = (2, 2, 2)) -> jnp.ndarray:
        lower, upper = self.parent._bbox
        basic_cube = grid_in_cube2(spacing=spacing,
                                   lower=jnp.array(lower),
                                   upper=jnp.array(upper))
        return basic_cube


class SamplingUniform(AbstractMethod, ExtendedNodeMixin):

    def __init__(self, parent: AbstractRegion):
        super(SamplingUniform, self).__init__()
        self.name = "sampling_uniform"
        self.parent = parent

    def __repr__(self):
        return f'sampling_uniform(N) -> xyz[N,3]'

    def __call__(self, rng_key: PRNGKeyArray, count: int) -> jnp.ndarray:
        lower, upper = self.parent._bbox
        basic_cube = uniform_in_cube(rng_key, count=count, lower=lower, upper=upper)
        return basic_cube


class SamplingGridWithShackle(AbstractMethod, ExtendedNodeMixin):

    def __init__(self, parent: AbstractRegion):
        super(SamplingGridWithShackle, self).__init__()
        self.name = "sampling_grid_with_shackle"
        self.parent = parent

    def __repr__(self):
        return f'sampling_grid_with_shackle(N) -> xyz[N,3]'

    def __call__(self, rng_key: PRNGKeyArray, spacing: (int, int, int) = (2, 2, 2), sigma=0.1) -> jnp.ndarray:
        assert (sigma > 0.00000001)
        lower, upper = self.parent._bbox
        shift_xyz = jax.random.normal(rng_key, shape=(3,)) * sigma
        basic_cube = grid_in_cube2(spacing=spacing,
                                   lower=jnp.array(lower),
                                   upper=jnp.array(upper)) + shift_xyz
        return basic_cube


class SamplingEachN(AbstractMethod, ExtendedNodeMixin):

    def __init__(self, parent: MeshRepr):
        super(SamplingEachN, self).__init__()
        self.name = "sampling_eachN"
        self.parent = parent

        self._points = None

    def __repr__(self):
        return f'sampling_eachN(count=N, step=1, shift=0) -> (ns_index[N], ns_xyz[N])'

    @property
    def points(self):
        if self._points is None:
            mesh = self.parent.surface_mesh2.mesh
            self._points = vtk_to_numpy(mesh.GetPoints().GetData())
        return self._points

    def unload_data(self):
        self._points = None

    def is_data_load(self):
        return (self._points is None)

    def __call__(self, count=1, step=1, shift=0) -> (jnp.ndarray, jnp.ndarray):
        index_set, array = take_each_n(self.points,
                                       count=count, step=step, shift=shift)
        ret_index_set = onp.zeros_like(index_set)
        ret_array = onp.zeros_like(array)
        for i in range(len(ret_index_set)):
            ret_index_set[i] = self.parent.index_physical2normed(index_set[i])
            ret_array[i] = self.parent.xyz_physical2normed(array[i])

        return ret_index_set, ret_array
