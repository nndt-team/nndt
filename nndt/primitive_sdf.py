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


class ConeSDF(AbstractSDF):
    """
    This is a cone geometric primitive.

    Args:
        center (tuple, optional): Center of the circle in the bottom of cone. (x, y, z). Defaults to (0.0, 0.0, 0.0).
        radius (float, optional): Radius of the circle in the bottom of cone. Defaults to 1.0.
        height (float, optional): Height of the cone. Get out from center and goes up along the Z-axis. Defaults to 1.0.
    """

    def __init__(self, center=(0.0, 0.0, 0.0), radius=1.0, height=1.0):
        self.center = center
        self.radius = radius
        self.height = height
        super(ConeSDF, self).__init__()

    @property
    def bbox(self) -> ((float, float, float), (float, float, float)):
        min_ = (
            (self.center[0] - self.radius),
            (self.center[1] - self.radius),
            (self.center[2]),
        )
        max_ = (
            (self.center[0] + self.radius),
            (self.center[1] + self.radius),
            (self.center[2] + self.height),
        )
        return min_, max_

    def _get_fun(self):
        center = self.center
        radius = self.radius

        def get_coeficients_of_linear_function_from_2_points(a, b):
            # Return the angle and bias for a linear function given by two points

            x1, x2, y1, y2 = a[0], b[0], a[1], b[1]
            b = 1 / (x2 - x1) * (y2 - y1)
            c = y1 - x1 / (x2 - x1)
            # y = lambda x: (x - x1) / (x2 - x1) * (y2 - y1) + y1
            return b, c

        def smallest_dist_binary_serch(a, b, point):
            # find smallest distance for point on line which starts at a point and ends at b point
            eps = 0.0000001
            left = a
            right = b
            while distance_between_2_points(left, right) > eps:
                dist_left = distance_between_2_points(left, point)
                dist_right = distance_between_2_points(right, point)
                if dist_left > dist_right:
                    left, right = right, left
                right = (right[0] / 2, right[1] / 2, right[2] / 2)
            return min(dist_right, dist_left)

        def distance_between_2_points(a, b):
            return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5

        def prim(x: float, y: float, z: float):
            if (
                z <= center[2]
                and center[0] - radius <= x <= center[0] + radius
                and center[1] - radius <= y <= center[1]
            ):
                # Case when the point is under the cone
                return abs(self.center[2] - z)

            # find angle and bias for linear function
            k, bias = get_coeficients_of_linear_function_from_2_points(
                (center[0], center[1]), (x, y)
            )
            angle = 0
            if k < 0:
                angle += pi
            angle = atan(k)
            # and find the point on the edge on the circle side of cone which is the closest to given point
            edge_point = (
                center[0] + radius * sin(angle),
                center[1] + radius * cos(angle),
                center[2],
            )

            heigh_point = (center[0], center[1], center[2] + self.height)
            # find distance from point to cone surface
            distance_to_cone_surface = smallest_dist_binary_serch(
                edge_point, heigh_point, (x, y, z)
            )
            if (
                center[0] - radius <= x <= center[0] + radius
                and center[1] - radius <= y <= center[1]
            ):
                # case when point is in cilinder shape
                return min(distance_to_cone_surface, abs(z - center[2]))

            # if it is outside of shape count only distance to surface
            return distance_to_cone_surface

        return prim
