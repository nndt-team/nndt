import pickle
import warnings
from typing import Optional

import jax
import jax.numpy as jnp
import vtk
from packaging import version
from pykdtree.kdtree import KDTree
from vtkmodules.util.numpy_support import vtk_to_numpy

import nndt
from nndt.space2.abstracts import AbstractLoader
from nndt.trainable_task import SimpleSDF


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
        with open(self.filepath, "r") as fl:
            self._text = fl.read()
        self.is_load = True

    def unload_data(self):
        self._text = None
        self.is_load = False

    def is_load(self) -> bool:
        return self.is_load


def _load_colors_from_obj(filepath):
    red = []
    green = []
    blue = []
    alpha = []

    with open(filepath, "r") as fl:
        for line in fl:
            if "v" in line:
                tokens = line.split(" ")
                if ("v" == tokens[0]) and (len(tokens) >= 7):
                    red.append(float(tokens[4].replace(",", ".")))
                    green.append(float(tokens[5].replace(",", ".")))
                    blue.append(float(tokens[6].replace(",", ".")))
                    alpha.append(1.0)

    red = jnp.array(red)
    green = jnp.array(green)
    blue = jnp.array(blue)
    alpha = jnp.array(alpha)

    return jnp.column_stack([red, green, blue, alpha])


def _load_colors_from_ply(filepath):
    red = []
    green = []
    blue = []
    alpha = []

    is_read_mode = False

    with open(filepath, "r") as fl:
        for line in fl:
            if "end_header" in line:
                is_read_mode = True
            if is_read_mode:
                tokens = line.split(" ")
                if len(tokens) >= 10:
                    red.append(float(tokens[6].replace(",", ".")))
                    green.append(float(tokens[7].replace(",", ".")))
                    blue.append(float(tokens[8].replace(",", ".")))
                    alpha.append(float(tokens[9].replace(",", ".")))

    red = jnp.array(red) / 255
    green = jnp.array(green) / 255
    blue = jnp.array(blue) / 255
    alpha = jnp.array(alpha) / 255

    return jnp.concatenate([red, green, blue, alpha], axis=1)


class MeshObjLoader(AbstractLoader):
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.is_load = False
        self._mesh = None
        self._points = None
        self._kdtree = None
        self._rgba = None

    def calc_bbox(self) -> ((float, float, float), (float, float, float)):
        Xmin, Xmax, Ymin, Ymax, Zmin, Zmax = self.mesh.GetBounds()
        return (Xmin, Ymin, Zmin), (Xmax, Ymax, Zmax)

    @property
    def mesh(self) -> vtk.vtkPolyData:
        if not self.is_load:
            self.load_data()
        return self._mesh

    @property
    def points(self) -> jnp.ndarray:
        if not self.is_load:
            self.load_data()
        return self._points

    @property
    def kdtree(self) -> KDTree:
        if not self.is_load:
            self.load_data()
        return self._kdtree

    @property
    def rgba(self) -> Optional[jnp.ndarray]:
        if not self.is_load:
            self.load_data()
        return self._rgba

    def load_data(self):
        reader = vtk.vtkOBJReader()
        reader.SetFileName(self.filepath)
        reader.Update()
        self._mesh = reader.GetOutput()
        onp_points = vtk_to_numpy(self._mesh.GetPoints().GetData())
        self._points = jnp.array(onp_points)
        self._kdtree = KDTree(onp_points)

        try:
            self._rgba = _load_colors_from_obj(self.filepath)
        except:
            warnings.warn("Colors cannot be loaded from mesh.")
            self._rgba = None

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
        self._sdt_threshold_level = 0.0

    def calc_bbox(self) -> ((float, float, float), (float, float, float)):
        mask_arr = self.sdt <= self._sdt_threshold_level
        Xmin = float(jnp.argmax(jnp.any(mask_arr, axis=(1, 2))))
        Ymin = float(jnp.argmax(jnp.any(mask_arr, axis=(0, 2))))
        Zmin = float(jnp.argmax(jnp.any(mask_arr, axis=(0, 1))))

        Xmax = float(
            self.sdt.shape[0] - jnp.argmax(jnp.any(mask_arr, axis=(1, 2))[::-1])
        )
        Ymax = float(
            self.sdt.shape[1] - jnp.argmax(jnp.any(mask_arr, axis=(0, 2))[::-1])
        )
        Zmax = float(
            self.sdt.shape[2] - jnp.argmax(jnp.any(mask_arr, axis=(0, 1))[::-1])
        )

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

        assert ps_xyz.ndim >= 1
        assert ps_xyz.shape[-1] == 3

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


class IR1Loader(AbstractLoader):
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.is_load = False

        self.json_ = None
        self.functions_ = None
        self.params_ = None
        self.bbox_ = None

    @property
    def json(self):
        if not self.is_load:
            self.load_data()
        return self.json_

    @property
    def functions(self):
        if not self.is_load:
            self.load_data()
        return self.functions_

    @property
    def params(self):
        if not self.is_load:
            self.load_data()
        return self.params_

    @property
    def bbox(self):
        if not self.is_load:
            self.load_data()
        return self.bbox_

    def load_data(self):
        with open(self.filepath, "rb") as input_file:
            self.json_ = pickle.load(input_file)
            version_ = self.json_["version"]
            trainable_task_ = self.json_["trainable_task"]
            repr_ = self.json_["repr"]

            history_loss_ = self.json_["history_loss"]
            params_ = self.json_["params"]
            bbox_ = self.json_["bbox"]

        if version.parse(nndt.__version__) < version.parse(version_):
            warnings.warn(
                "Loaded neural network was created on earlier version of NNDT!"
            )

        task = SimpleSDF(**trainable_task_)
        rng = jax.random.PRNGKey(42)
        _, self.F = task.init_and_functions(rng)

        self.functions_ = self.F
        self.params_ = params_
        self.bbox_ = bbox_

        self.is_load = True

    def unload_data(self):
        self.json_ = None
        self.functions_ = None
        self.params_ = None
        self.bbox_ = None

    def is_load(self) -> bool:
        return self.is_load


DICT_LOADERTYPE_CLASS = {
    "txt": TXTLoader,
    "sdt": SDTLoader,
    "mesh_obj": MeshObjLoader,
    "implicit_ir1": IR1Loader,
    "undefined": EmptyLoader,
}
DICT_CLASS_LOADERTYPE = {(v, k) for k, v in DICT_LOADERTYPE_CLASS.items()}
