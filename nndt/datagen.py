import jax
import jax.numpy as jnp

import nndt


def rotation_matrix(yaw, pitch, roll):
    Rz = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw), 0.],
                    [jnp.sin(yaw), jnp.cos(yaw), 0.],
                    [0., 0., 1.]])
    Ry = jnp.array([[jnp.cos(pitch), 0, jnp.sin(pitch)],
                    [0, 1, 0],
                    [-jnp.sin(pitch), 0., jnp.cos(pitch)]])
    Rx = jnp.array([[1., 0., 0.],
                    [0., jnp.cos(roll), -jnp.sin(roll)],
                    [0., jnp.sin(roll), jnp.cos(roll)]])

    return Rz @ Ry @ Rx


def scale_and_rotate(xyz, scale=(1., 1., 1.), rotate=(0., 0., 0.)):
    assert (xyz.shape[-1] == 3)
    M = rotation_matrix(*rotate)
    scale = jnp.array(scale)

    ret_shape = xyz.shape
    xyz = xyz.reshape((-1, 3))
    xyz = (M @ (xyz * scale).T).T

    xyz = xyz.reshape(ret_shape)
    return xyz


class DataGenerator:

    def __init__(self,
                 node,
                 cube_spacing=(16, 16, 16),
                 cube_scale=1.,
                 count=33,
                 step=77,
                 sigma=0.09,
                 shift_mul=4,
                 rand_scale_sub=0.1, augment=True,
                 rotation=1):
        self.node = node
        self.count = count
        self.step = step
        self.sigma = sigma
        self.shift_mul = shift_mul
        self.ns_cube = nndt.math_core.grid_in_cube(spacing=cube_spacing,
                                                   scale=cube_scale,
                                                   center_shift=(0., 0., 0.))
        self.rotation = rotation

        self.rand_scale_sub = rand_scale_sub
        self.augment = augment

    def _process_one_model(self, key, obj, epoch, batch_color_class: list, batch_sdt: list):

        ind, xyz = obj.sampling_eachN_from_mesh(count=self.count,
                                                step=self.step,
                                                shift=self.shift_mul * epoch)
        key, subkey = jax.random.split(key)
        if self.augment:
            xyz = xyz + self.sigma * jax.random.normal(subkey, shape=xyz.shape)
        else:
            pass
        rgba = obj.surface_xyz2rgba(xyz)
        color_class = jnp.argmax(rgba[:, 0:3], axis=1)
        batch_color_class.append(color_class)

        if self.augment:
            scale_list = jax.random.uniform(subkey, shape=(len(xyz), 3,),
                                            minval=1. - self.rand_scale_sub,
                                            maxval=1. + self.rand_scale_sub)
            rotate_list = jax.random.uniform(subkey, shape=(len(xyz), 3,),
                                             minval=-self.rotation,
                                             maxval=self.rotation)

        else:
            scale_list = [(1., 1., 1.)] * len(xyz)
            rotate_list = [(0., 0., 0.)] * len(xyz)

        batch_sdtlocal = []
        for idx, point in enumerate(xyz):
            scale = scale_list[idx]
            rotate = rotate_list[idx]
            ns_cube = scale_and_rotate(self.ns_cube, scale=scale, rotate=rotate)
            ns_cube = ns_cube + point
            sdtlocal = obj.surface_xyz2sdt(ns_cube)

            batch_sdtlocal.append(sdtlocal)
        batch_sdt.append(jnp.stack(batch_sdtlocal, axis=0))

        return batch_color_class, batch_sdt, subkey

    def get(self, key, epoch, index=None):
        batch_color_class = []
        batch_sdt = []

        if index is None:
            for obj in self.node:
                batch_color_class, batch_sdt, subkey = self._process_one_model(key, obj, epoch, batch_color_class,
                                                                               batch_sdt)
        else:
            obj = self.node[index]
            batch_color_class, batch_sdt, subkey = self._process_one_model(key, obj, epoch, batch_color_class,
                                                                           batch_sdt)

        batch_color_class = jnp.concatenate(batch_color_class, axis=0)
        batch_sdt = jnp.concatenate(batch_sdt, axis=0)

        return batch_sdt, batch_color_class, subkey
