from abc import abstractmethod

import jax.numpy as jnp
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy


class AbstractLoader():

    def calc_bbox(self) -> ((float, float, float), (float, float, float)):
        return (0., 0., 0.), (0., 0., 0.)

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def unload_data(self):
        pass

    @abstractmethod
    def is_load(self) -> bool:
        pass


class EmptyLoader(AbstractLoader):

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.is_load = False

    def load_data(self):
        self.is_load = True

    def unload_data(self):
        self.is_load = False

    def is_load(self) -> bool:
        return self.is_load


class TXTLoader(AbstractLoader):

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.is_load = False
        self._text = None

    @property
    def text(self):
        if not self.is_load:
            self.load_data()
        return self._text

    def load_data(self):
        with open(self.filepath, 'r') as fl:
            self._text = fl.read()
        self.is_load = True

    def unload_data(self):
        self._text = None
        self.is_load = False

    def is_load(self) -> bool:
        return self.is_load


class MeshObjLoader(AbstractLoader):
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.is_load = False
        self._mesh = None
        self._points = None

    def calc_bbox(self) -> ((float, float, float), (float, float, float)):
        Xmin, Xmax, Ymin, Ymax, Zmin, Zmax = self.mesh.GetBounds()
        return (Xmin, Ymin, Zmin), (Xmax, Ymax, Zmax)

    @property
    def mesh(self):
        if not self.is_load:
            self.load_data()
        return self._mesh

    @property
    def points(self):
        if not self.is_load:
            self.load_data()
        return self._points

    def load_data(self):
        reader = vtk.vtkOBJReader()
        reader.SetFileName(self.filepath)
        reader.Update()
        self._mesh = reader.GetOutput()
        self._points = vtk_to_numpy(self._mesh.GetPoints().GetData())
        self.is_load = True

    def unload_data(self):
        self._mesh = None
        self.is_load = False

    def is_load(self) -> bool:
        return self.is_load


class SDTLoader(AbstractLoader):
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.is_load = False
        self._sdt = None
        self._sdt_threshold_level = 0.

    def calc_bbox(self) -> ((float, float, float), (float, float, float)):
        mask_arr = (self.sdt <= self._sdt_threshold_level)
        Xmin = float(jnp.argmax(jnp.any(mask_arr, axis=(1, 2))))
        Ymin = float(jnp.argmax(jnp.any(mask_arr, axis=(0, 2))))
        Zmin = float(jnp.argmax(jnp.any(mask_arr, axis=(0, 1))))

        Xmax = float(self.sdt.shape[0] - jnp.argmax(jnp.any(mask_arr, axis=(1, 2))[::-1]))
        Ymax = float(self.sdt.shape[1] - jnp.argmax(jnp.any(mask_arr, axis=(0, 2))[::-1]))
        Zmax = float(self.sdt.shape[2] - jnp.argmax(jnp.any(mask_arr, axis=(0, 1))[::-1]))

        return (Xmin, Ymin, Zmin), (Xmax, Ymax, Zmax)

    @property
    def sdt(self):
        if not self.is_load:
            self.load_data()
        return self._sdt

    def load_data(self):
        self._sdt = jnp.load(self.filepath)
        self.is_load = True

    def request(self, ps_xyz: jnp.ndarray) -> jnp.ndarray:

        assert (ps_xyz.ndim >= 1)
        assert (ps_xyz.shape[-1] == 3)

        if ps_xyz.ndim == 1:
            p_array_ = ps_xyz[jnp.newaxis, :]
        else:
            p_array_ = ps_xyz

        p_array_ = p_array_.reshape((-1, 3))

        x = jnp.clip(p_array_[:, 0], 0, self.sdt.shape[0] - 1).astype(int)
        y = jnp.clip(p_array_[:, 1], 0, self.sdt.shape[1] - 1).astype(int)
        z = jnp.clip(p_array_[:, 2], 0, self.sdt.shape[2] - 1).astype(int)

        result = self.sdt[x, y, z]

        ret_shape = list(ps_xyz.shape)
        ret_shape[-1] = 1

        result = result.reshape(ret_shape)

        return result

    def unload_data(self):
        self._sdt = None
        self.is_load = False

    def is_load(self) -> bool:
        return self.is_load


DICT_LOADERTYPE_CLASS = {'txt': TXTLoader,
                         'sdt': SDTLoader,
                         'mesh_obj': MeshObjLoader,
                         'undefined': EmptyLoader}
DICT_CLASS_LOADERTYPE = {(v, k) for k, v in DICT_LOADERTYPE_CLASS.items()}
