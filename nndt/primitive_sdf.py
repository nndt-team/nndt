from abc import ABC

import jax


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


class AbstractSDF(ABC):

    @property
    def bbox(self) -> ((float, float, float), (float, float, float)):
        return (0., 0., 0.), (0., 0., 0.)


class SphereSDF(AbstractSDF):

    def __init__(self, center=(0., 0., 0.), radius=1.):
        assert (radius > 0.)
        self.center = center
        self.radius = radius
        tpl = fun2vec_and_grad(self.get_pure)
        self.vec_prim = tpl[0]
        self.vec_prim_x = tpl[1]
        self.vec_prim_y = tpl[2]
        self.vec_prim_z = tpl[3]

    @property
    def bbox(self) -> ((float, float, float), (float, float, float)):
        min_ = (self.center[0] - self.radius), (self.center[1] - self.radius), (self.center[2] - self.radius)
        max_ = (self.center[0] + self.radius), (self.center[1] + self.radius), (self.center[2] + self.radius)
        return min_, max_

    def get_pure(self):
        center = self.center
        radius = self.radius

        def prim(x: float, y: float, z: float):
            sdf = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2 - radius ** 2
            return sdf

        return prim
