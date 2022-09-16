import numpy as onp
import os
import vtk
from abc import abstractmethod
from pathlib import Path


class UnloadMixin:

    @abstractmethod
    def unload_data(self):
        pass

    @abstractmethod
    def is_data_load(self) -> bool:
        pass


class SurfaceMesh(UnloadMixin):

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._mesh = None

    @property
    def mesh(self):
        if self._mesh is None:
            self._mesh = self._load_mesh()
        return self._mesh

    def unload_data(self):
        self._mesh = None

    def is_data_load(self):
        return (self._mesh is not None)

    def _load_mesh(self):

        self.file = Path(self.filepath)
        self.file.resolve(strict=True)
        _, file_extension = os.path.splitext(self.file)

        if file_extension == ".ply":
            reader = vtk.vtkPLYReader()
        elif file_extension == ".obj":
            reader = vtk.vtkOBJReader()
        else:
            raise NotImplementedError(f"Mesh with {file_extension} is not supported. "
                                      f"The following file types are available: .obj, .ply.")
        reader.SetFileName(str(self.file))
        reader.Update()
        _mesh = reader.GetOutput()

        return _mesh


class SDTExplicitArray(UnloadMixin):

    def __init__(self, filepath: str):
        self.filepath = filepath

        self.file = Path(filepath)
        self.file.resolve(strict=True)
        _, file_extension = os.path.splitext(self.file)
        assert ("npy" in file_extension)

        self._sdt = None

    @property
    def sdt(self):
        if self._sdt is None:
            self._sdt = self._load_sdt()
        return self._sdt

    def unload_data(self):
        self._sdt = None

    def is_data_load(self):
        return (self._sdt is not None)

    def _load_sdt(self):
        return onp.load(self.filepath)

    def min_bbox(self):
        mask_arr = (self.sdt <= 0.)
        Xmin = onp.argmax(onp.any(mask_arr, axis=(1, 2)))
        Ymin = onp.argmax(onp.any(mask_arr, axis=(0, 2)))
        Zmin = onp.argmax(onp.any(mask_arr, axis=(0, 1)))

        Xmax = self.sdt.shape[0] - onp.argmax(onp.any(mask_arr, axis=(1, 2))[::-1])
        Ymax = self.sdt.shape[1] - onp.argmax(onp.any(mask_arr, axis=(0, 2))[::-1])
        Zmax = self.sdt.shape[2] - onp.argmax(onp.any(mask_arr, axis=(0, 1))[::-1])

        return Xmin, Xmax, Ymin, Ymax, Zmin, Zmax

    def request(self, ps_xyz: onp.ndarray) -> onp.ndarray:

        assert (ps_xyz.ndim >= 1)
        assert (ps_xyz.shape[-1] == 3)

        if ps_xyz.ndim == 1:
            p_array_ = ps_xyz[onp.newaxis, :]
        else:
            p_array_ = ps_xyz

        p_array_ = p_array_.reshape((-1, 3))

        x = onp.clip(p_array_[:, 0], 0, self.sdt.shape[0] - 1).astype(int)
        y = onp.clip(p_array_[:, 1], 0, self.sdt.shape[1] - 1).astype(int)
        z = onp.clip(p_array_[:, 2], 0, self.sdt.shape[2] - 1).astype(int)

        result = self.sdt[x, y, z]

        ret_shape = list(ps_xyz.shape)
        ret_shape[-1] = 1

        result = result.reshape(ret_shape)

        return result
