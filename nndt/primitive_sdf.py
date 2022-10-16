from abc import abstractmethod
from typing import Callable

import jax
import jax.numpy as jnp


def sdf_primitive_sphere(center=(0., 0., 0.), radius=1.):
    """

    Parameters
    ----------
    center : tuple, optional
        Coordinates of center x, y, z (defaults is (0., 0., 0.))
    radius : float, optional
        Radius of sphere (defaults is 1.)

    Returns
    -------
    Set of jax.vmap
        description smth, x, y, z
    """

    def prim(x: float, y: float, z: float):
        sdf = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2 - radius ** 2
        return sdf

    vec_prim = jax.vmap(prim)

    prim_x = jax.grad(prim, argnums=0)
    prim_y = jax.grad(prim, argnums=1)
    prim_z = jax.grad(prim, argnums=2)

    vec_prim_x = jax.vmap(prim_x)
    vec_prim_y = jax.vmap(prim_y)
    vec_prim_z = jax.vmap(prim_z)

    return vec_prim, vec_prim_x, vec_prim_y, vec_prim_z


def fun2vec_and_grad(prim):
    vec_prim = jax.vmap(prim)

    prim_x = jax.grad(prim, argnums=0)
    prim_y = jax.grad(prim, argnums=1)
    prim_z = jax.grad(prim, argnums=2)

    vec_prim_x = jax.vmap(prim_x)
    vec_prim_y = jax.vmap(prim_y)
    vec_prim_z = jax.vmap(prim_z)

    return vec_prim, vec_prim_x, vec_prim_y, vec_prim_z


class AbstractSDF:

    def __init__(self):
        tpl = fun2vec_and_grad(self.get_fun())
        self._vec_fun = tpl[0]
        self._vec_fun_x = tpl[1]
        self._vec_fun_y = tpl[2]
        self._vec_fun_z = tpl[3]

    @abstractmethod
    def get_fun(self):
        pass

    @property
    @abstractmethod
    def bbox(self) -> ((float, float, float), (float, float, float)):
        return (0., 0., 0.), (0., 0., 0.)

    @property
    def vec_fun(self) -> Callable:
        return self._vec_fun

    @property
    def vec_fun_dx(self) -> Callable:
        return self._vec_fun_x

    @property
    def vec_fun_dy(self) -> Callable:
        return self._vec_fun_y

    @property
    def vec_fun_dz(self) -> Callable:
        return self._vec_fun_z

    def request(self, ps_xyz: jnp.ndarray) -> jnp.ndarray:
        assert (ps_xyz.shape[-1] == 3)

        ret_shape = list(ps_xyz.shape)
        ret_shape[-1] = 1
        ret_shape = tuple(ret_shape)

        x = ps_xyz[..., 0].flatten()
        y = ps_xyz[..., 1].flatten()
        z = ps_xyz[..., 2].flatten()
        dist = self._vec_fun(x, y, z)
        dist = dist.reshape(ret_shape)

        return dist


class SphereSDF(AbstractSDF):

    def __init__(self, center=(0., 0., 0.), radius=1.):
        assert (radius > 0.)
        self.center = center
        self.radius = radius
        super(SphereSDF, self).__init__()

    @property
    def bbox(self) -> ((float, float, float), (float, float, float)):
        min_ = (self.center[0] - self.radius), (self.center[1] - self.radius), (self.center[2] - self.radius)
        max_ = (self.center[0] + self.radius), (self.center[1] + self.radius), (self.center[2] + self.radius)
        return min_, max_

    def get_fun(self):
        center = self.center
        radius = self.radius

        def prim(x: float, y: float, z: float):
            sdf = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2 - radius ** 2
            return sdf

        return prim
