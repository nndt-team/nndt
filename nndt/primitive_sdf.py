from abc import abstractmethod
from math import sqrt
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


class BoxSDF(AbstractSDF):
    """
    This is a box geometrical primitive. Sides are parallel with XYZ axis.
    """

    def __init__(self, first_vertex=(0.0, 0.0, 0.0), opposite_vertex=(0.0, 0.0, 0.0)):
        """
        This is a box geometrical primitive. Sides are parallel with XYZ axis

        :param first_vertex: vertex of a box
        :param opposite_vertex: the opposite vertex for the first_vertex.
        opposite_vertex is the only vertex of a box that is not on the same plane
        with first_vertex
        """
        self.first_vertex = first_vertex
        self.opposite_vertex = opposite_vertex
        super(BoxSDF, self).__init__()

    @property
    def bbox(self) -> ((float, float, float), (float, float, float)):
        return self.first_vertex, self.opposite_vertex

    def _get_fun(self):
        first_vertex = self.first_vertex
        opposite_vertex = self.opposite_vertex

        def prim(x: float, y: float, z: float):
            min_xyz = (
                min(first_vertex[0], opposite_vertex[0]),
                min(first_vertex[1], opposite_vertex[1]),
                min(first_vertex[2], opposite_vertex[2]),
            )
            max_xyz = (
                max(first_vertex[0], opposite_vertex[0]),
                max(first_vertex[1], opposite_vertex[1]),
                max(first_vertex[2], opposite_vertex[2]),
            )
            xyz_on_box = ()
            dist_to_planes_xyz = ()
            for i in range(3):
                on_box = (
                    jnp.where(x < min_xyz[i], min_xyz[i], jnp.array(())),
                    jnp.where((min_xyz[i] <= x) & (x <= max_xyz[i]), x, jnp.array(())),
                    jnp.where(max_xyz[i] < x, max_xyz[i], jnp.array(())),
                )
                for j in range(3):
                    if type(on_box[j]) == float:
                        xyz_on_box += on_box[j]
                dist_to_planes_xyz += jnp.where(
                    (min_xyz[i] <= x) & (x <= max_xyz[i]),
                    min(jnp.abs(x - min_xyz[i]), jnp.abs(x - max_xyz[i])),
                    None,
                )

            if None not in dist_to_planes_xyz:
                return -1 * min(dist_to_planes_xyz)

            return sqrt(
                (xyz_on_box[0] - x) ** 2
                + (xyz_on_box[1] - y) ** 2
                + (xyz_on_box[2] - z) ** 2
            )

        return prim
