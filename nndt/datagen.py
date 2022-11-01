from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from jax.random import KeyArray

import nndt
from nndt.math_core import rotation_matrix as _rotation_matrix
from nndt.math_core import scale_xyz as _scale_xyz
from nndt.trainable_task import ApproximateSDF


def _rotate_xyz(cube, M):
    M = _rotation_matrix(M[0], M[1], M[2]).T
    return cube @ M


def _shift_xyz(cube, shift):
    return cube + shift


def _scale_rotate_shift(
    cube, scale=(1.0, 1.0, 1.0), rotation=(0.0, 0.0, 0.0), shift=(0.0, 0.0, 0.0)
):
    cube = _scale_xyz(cube, scale)
    cube = _rotate_xyz(cube, rotation)
    cube = _shift_xyz(cube, shift)
    return cube


_vec_scale_rotate_shift = jax.jit(jax.vmap(_scale_rotate_shift, (0, 0, 0, 0)))


class AbstractDatagen(ABC):
    @abstractmethod
    def get(self, key: KeyArray, epoch: int, **kwargs):
        pass


class DataGenForShapeRegression(AbstractDatagen):
    def __init__(self, node, spacing=(4.0, 4.0, 4.0), shift_sigma=0.05, augment=True):
        self.node = node
        self.spacing = spacing

        self.augment = augment
        if self.augment:
            self.shift_sigma = shift_sigma
        else:
            self.shift_sigma = 0.0

    def get(self, key, epoch, **kwargs):
        num_of_obj = len(self.node)

        X_list = []
        Y_list = []
        Z_list = []
        T_list = []
        P_list = []
        SDF_list = []

        xyz = self.node.sampling_grid_with_noise(
            key, spacing=self.spacing, sigma=self.shift_sigma
        )
        xyz_flat = xyz.reshape((-1, 3))

        for code, patient in enumerate(self.node):
            sdf_flat = jnp.squeeze(patient.surface_xyz2sdt(xyz_flat))

            p_array = jnp.array(jnp.zeros((sdf_flat.shape[0], num_of_obj)))
            p_array = p_array.at[:, code].set(1.0)

            X_list.append(xyz_flat[:, 0])
            Y_list.append(xyz_flat[:, 1])
            Z_list.append(xyz_flat[:, 2])
            T_list.append(jnp.zeros(sdf_flat.shape[0]))
            P_list.append(p_array)
            SDF_list.append(sdf_flat)

        DATA = ApproximateSDF.DATA(
            X=jnp.concatenate(X_list, axis=0),
            Y=jnp.concatenate(Y_list, axis=0),
            Z=jnp.concatenate(Z_list, axis=0),
            T=jnp.concatenate(T_list, axis=0),
            P=jnp.concatenate(P_list, axis=0),
            SDF=jnp.concatenate(SDF_list, axis=0),
        )

        return DATA


class DataGenForSegmentation(AbstractDatagen):
    def __init__(
        self,
        node,
        cube_spacing=(16, 16, 16),
        cube_scale=1.0,
        count=33,
        step=77,
        shift_sigma=0.09,
        scale_range=0.03,
        rotate_angle=1.0,
        shift_mul=4,
        augment=True,
    ):
        self.node = node
        self.count = count
        self.step = step
        self.augment = augment
        self.shift_mul = shift_mul

        if self.augment:
            self.shift_sigma = shift_sigma
            self.scale_range = scale_range
            self.rotate_angle = rotate_angle
        else:
            self.shift_sigma = 0.0
            self.scale_range = 0.0
            self.rotate_angle = 0.0

        one_cube = nndt.math_core.grid_in_cube(
            spacing=cube_spacing, scale=cube_scale, center_shift=(0.0, 0.0, 0.0)
        )
        self.basic_cube = jnp.repeat(one_cube[jnp.newaxis, ...], self.count, axis=0)

    def _process_one_model(self, key, obj, epoch):
        key, subkey = jax.random.split(key)
        ind, xyz = obj.sampling_eachN_from_mesh(
            count=self.count, step=self.step, shift=self.shift_mul * epoch
        )
        key, subkey = jax.random.split(key)
        scale = jax.random.uniform(
            subkey,
            shape=(self.count, 3),
            minval=1.0 - self.scale_range,
            maxval=1.0 + self.scale_range,
        )
        key, subkey = jax.random.split(key)
        rotate = jax.random.uniform(
            subkey,
            shape=(self.count, 3),
            minval=0.0 - self.rotate_angle,
            maxval=0.0 + self.rotate_angle,
        )
        key, subkey = jax.random.split(key)
        shift = jax.random.normal(subkey, shape=(self.count, 3)) * self.shift_sigma

        new_pos_xyz = xyz + shift
        _xyz_cube = _vec_scale_rotate_shift(self.basic_cube, scale, rotate, new_pos_xyz)

        rgba = obj.surface_xyz2rgba(new_pos_xyz)
        color_class = jnp.argmax(rgba[:, 0:3], axis=1)
        sdt = jax.jit(obj.surface_xyz2sdt)(_xyz_cube)

        return subkey, color_class, sdt

    def get(self, key, epoch, index=None):
        batch_color_class = []
        batch_sdt = []
        lst = [index] if index is not None else range(len(self.node))

        key, subkey = jax.random.split(key)

        for i in lst:
            obj = self.node[i]
            subkey, color_class, sdt = self._process_one_model(subkey, obj, epoch)
            batch_color_class.append(color_class)
            batch_sdt.append(sdt)

        batch_color_class = jnp.concatenate(batch_color_class, axis=0)
        batch_sdt = jnp.concatenate(batch_sdt, axis=0)

        return batch_sdt, batch_color_class
