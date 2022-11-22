from abc import abstractmethod
from typing import Callable

import jax
import jax.numpy as jnp


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
        self._fun = self._get_fun()
        tpl = fun2vec_and_grad(self._fun)
        self._vec_fun = tpl[0]
        self._vec_fun_x = tpl[1]
        self._vec_fun_y = tpl[2]
        self._vec_fun_z = tpl[3]

    @abstractmethod
    def _get_fun(self):
        pass

    @property
    @abstractmethod
    def bbox(self) -> ((float, float, float), (float, float, float)):
        """
        Return the minimal bounding box around the implicitly defined object.
        :return: (X_min, Y_min, Z_min) , (X_max, Y_max, Z_max)
        """
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    @property
    def fun(self) -> Callable:
        """
        Get the SDF function in scalar form
        :return: `f(x,y,z) = distance`
        """
        return self._fun

    @property
    def vec_fun(self) -> Callable:
        """
        Get the SDF function in vector form. Vectorization is performed along the zero axis.
        :return: `f(vec_x, vec_y, vec_z) = vec_distance`
        """
        return self._vec_fun

    @property
    def vec_fun_dx(self) -> Callable:
        """
        Get the gradient of the SDF function over the X-axis. Vectorization is performed along the zero axis.
        :return: `df/dx(vec_x, vec_y, vec_z)`
        """
        return self._vec_fun_x

    @property
    def vec_fun_dy(self) -> Callable:
        """
        Get the gradient of the SDF function over the Y-axis. Vectorization is performed along the zero axis.
        :return: `df/dy(vec_x, vec_y, vec_z)`
        """
        return self._vec_fun_y

    @property
    def vec_fun_dz(self) -> Callable:
        """
        Get the gradient of the SDF function over the Z-axis. Vectorization is performed along the zero axis.
        :return: `df/dz(vec_x, vec_y, vec_z)`
        """
        return self._vec_fun_z

    def request(self, ps_xyz: jnp.ndarray) -> jnp.ndarray:
        """
        Get SDF values for the requested location on the physical space.
        :return: distance values
        """
        assert ps_xyz.shape[-1] == 3

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
    """
    This is a sphere geometrical primitive.
    """

    def __init__(self, center=(0.0, 0.0, 0.0), radius=1.0):
        """
        This is a sphere geometrical primitive.

        :param center: center of the sphere
        :param radius: radius of the sphere
        """
        assert radius > 0.0
        self.center = center
        self.radius = radius
        super(SphereSDF, self).__init__()

    @property
    def bbox(self) -> ((float, float, float), (float, float, float)):
        min_ = (
            (self.center[0] - self.radius),
            (self.center[1] - self.radius),
            (self.center[2] - self.radius),
        )
        max_ = (
            (self.center[0] + self.radius),
            (self.center[1] + self.radius),
            (self.center[2] + self.radius),
        )
        return min_, max_

    def _get_fun(self):
        center = self.center
        radius = self.radius

        def prim(x: float, y: float, z: float):
            sdf = (
                (x - center[0]) ** 2
                + (y - center[1]) ** 2
                + (z - center[2]) ** 2
                - radius**2
            )
            return sdf

        return prim

class CylinderSDF(AbstractSDF):
    def __init__(self, center=(0.0, 0.0, 0.0), radius=1.0, height=1.0):
        assert radius > 0.0 and height > 0.0
        self.center = center
        self.radius = radius
        self.height = height
        super(CylinderSDF, self).__init__()

    @property
    def bbox(self) -> ((float, float, float), (float, float, float)):
        min_ = (
            (self.center[0] - self.radius),
            (self.center[1] - self.height / 2),
            (self.center[2] - self.radius),
        )
        max_ = (
            (self.center[0] + self.radius),
            (self.center[1] + self.height / 2),
            (self.center[2] + self.radius),
        )
        return min_, max_

    def _get_fun(self):
        center = self.center
        radius = self.radius
        height = self.height

        def prim(x: float, y: float, z: float):
            min_, max_ = self.bbox

            xz_dist = jnp.sqrt((x - center[0]) ** 2 + (z - center[2]) ** 2)
            y_dist = jnp.abs(y - center[1])

            if min_[1] <= y <= max_[1]:
                if xz_dist > radius:
                    sdf = xz_dist - radius
                else:
                    sdf = jnp.max(xz_dist - radius, y_dist - height/2)
            else:
                if xz_dist <= radius:
                    sdf = y_dist - height/2
                else:
                    sdf = jnp.sqrt((xz_dist - radius) ** 2 + (y_dist - height/2) ** 2)

            return sdf

        return prim