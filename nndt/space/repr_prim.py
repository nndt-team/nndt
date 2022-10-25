import jax.numpy as jnp

from nndt.primitive_sdf import sdf_primitive_sphere
from nndt.space.abstracts import *
from nndt.space.sources import SphereSDFSource


class SphereSDF(AbstractRegion, ExtendedNodeMixin):
    def __init__(self, parent: SphereSDFSource, name=""):
        self.center = center = parent.center
        self.radius = radius = parent.radius
        super(SphereSDF, self).__init__(
            _ndim=3,
            _bbox=(
                (center[0] - radius, center[1] - radius, center[2] - radius),
                (center[0] + radius, center[1] + radius, center[2] + radius),
            ),
            name=name,
        )
        self.name = name
        self.parent = parent

        self._print_color = Fore.GREEN

        fun = sdf_primitive_sphere(center=center, radius=radius)
        self.vec_prim, self.vec_prim_x, self.vec_prim_y, self.vec_prim_z = fun


class SphereSDF_Xyz2SDT(AbstractMethod, ExtendedNodeMixin):
    def __init__(self, parent: SphereSDF):
        super(SphereSDF_Xyz2SDT, self).__init__()
        self.name = "xyz2sdt"
        self.parent = parent

    def __repr__(self):
        return f"xyz2sdt(ns_xyz[...,3]) -> ns_sdt[...,1]"

    def __call__(self, ns_xyz: jnp.ndarray) -> jnp.ndarray:
        ns_sdt = self.parent.vec_prim(ns_xyz[:, 0], ns_xyz[:, 1], ns_xyz[:, 2])
        return ns_sdt[..., jnp.newaxis]


class SphereSDF_PureSDF(AbstractMethod, ExtendedNodeMixin):
    def __init__(self, parent: SphereSDF):
        super(SphereSDF_PureSDF, self).__init__()
        self.name = "pure_sdf"
        self.parent = parent

    def __repr__(self):
        return f"pure_sdf() -> F(x,y,z)=sdf"

    def __call__(self) -> Callable[[float, float, float], float]:
        x0, y0, z0 = self.parent.center
        r = self.parent.radius

        def prim(x: float, y: float, z: float):
            sdf = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2 - r**2
            return sdf

        return prim
