import jax.numpy as jnp

from nndt.math_core import grid_in_cube
from nndt.space.abstracts import *
from nndt.space.sources import SDTSource
from nndt.space.utils import downup_update_bbox
from nndt.space.vtk_wrappers import *


class SDTRepr(AbstractRegion, ExtendedNodeMixin, UnloadMixin):
    MAGIC_CORRECTION = 0.503  # This is absolutely magic coefficient that reduce error between bboxes to 0.49075 mm

    def __init__(self, parent: AbstractSource, sdt_explicit_array2: SDTExplicitArray,
                 physical_center: (float, float, float),
                 physical_bbox: ((float, float, float), (float, float, float)),
                 normed_center: (float, float, float),
                 normed_bbox: ((float, float, float), (float, float, float)),
                 scale_physical2normed: float,
                 _ndim=3,
                 _scale=1.,
                 name=""):
        super(SDTRepr, self).__init__(_ndim=_ndim,
                                      _bbox=normed_bbox,
                                      name=name)
        self.name = name
        self.parent = parent
        self._sdt_explicit_array2 = sdt_explicit_array2

        self.physical_center = onp.array(physical_center)
        self.physical_bbox = physical_bbox
        self.normed_center = onp.array(normed_center)
        self.normed_bbox = normed_bbox

        self.scale_physical2normed = scale_physical2normed
        self._print_color = Fore.GREEN

    def unload_mesh(self):
        self._sdt_explicit_array2.unload_data()

    def is_data_load(self):
        return self._sdt_explicit_array2.is_data_load()

    def ps_xyz2sdt(self, ps_xyz: onp.ndarray) -> onp.ndarray:
        ps_sdt = self._sdt_explicit_array2.request(ps_xyz)
        return ps_sdt

    def ns_xyz2sdt(self, ns_xyz: onp.ndarray) -> onp.ndarray:
        ps_xyz = (ns_xyz - self.normed_center) * self.scale_physical2normed + self.physical_center
        ps_sdt = self._sdt_explicit_array2.request(ps_xyz)
        ns_sdt = ps_sdt / self.scale_physical2normed
        return ns_sdt

    @classmethod
    def load_mesh_and_bring_to_center(cls, source: SDTSource,
                                      padding_physical=(10, 10, 10),
                                      scale_physical2normed=50):
        sdt_explicit_array2 = SDTExplicitArray(source.filepath)
        Xmin, Xmax, Ymin, Ymax, Zmin, Zmax = sdt_explicit_array2.min_bbox()
        sdt_explicit_array2.unload_data()

        normed_center = (0., 0., 0.)
        physical_bbox = ((Xmin - padding_physical[0] + SDTRepr.MAGIC_CORRECTION,
                          Ymin - padding_physical[1] + SDTRepr.MAGIC_CORRECTION,
                          Zmin - padding_physical[2] + SDTRepr.MAGIC_CORRECTION),
                         (Xmax + padding_physical[0] - SDTRepr.MAGIC_CORRECTION,
                          Ymax + padding_physical[1] - SDTRepr.MAGIC_CORRECTION,
                          Zmax + padding_physical[2] - SDTRepr.MAGIC_CORRECTION))
        physical_center = (physical_bbox[0][0] + (physical_bbox[1][0] - physical_bbox[0][0]) / 2.,
                           physical_bbox[0][1] + (physical_bbox[1][1] - physical_bbox[0][1]) / 2.,
                           physical_bbox[0][2] + (physical_bbox[1][2] - physical_bbox[0][2]) / 2.)

        scale_physical2normed = scale_physical2normed

        normed_bbox = (((physical_bbox[0][0] - physical_center[0]) / scale_physical2normed + normed_center[0],
                        (physical_bbox[0][1] - physical_center[1]) / scale_physical2normed + normed_center[1],
                        (physical_bbox[0][2] - physical_center[2]) / scale_physical2normed + normed_center[2]),
                       ((physical_bbox[1][0] - physical_center[0]) / scale_physical2normed + normed_center[0],
                        (physical_bbox[1][1] - physical_center[1]) / scale_physical2normed + normed_center[1],
                        (physical_bbox[1][2] - physical_center[2]) / scale_physical2normed + normed_center[2]))

        repr = SDTRepr(source, sdt_explicit_array2,
                       physical_center,
                       physical_bbox,
                       normed_center,
                       normed_bbox,
                       scale_physical2normed,
                       name="repr")

        downup_update_bbox(repr)

        return repr


class Xyz2SDT(AbstractMethod, ExtendedNodeMixin):

    def __init__(self, parent: SDTRepr):
        super(Xyz2SDT, self).__init__()
        self.name = "xyz2sdt"
        self.parent = parent

    def __repr__(self):
        return f'xyz2sdt(ns_xyz[...,3]) -> ns_sdt[...,1]'

    def __call__(self, ns_xyz: jnp.ndarray) -> jnp.ndarray:
        ns_sdt = self.parent.ns_xyz2sdt(ns_xyz)
        return ns_sdt


class Xyz2LocalSDT(AbstractMethod, ExtendedNodeMixin):

    def __init__(self, parent: SDTRepr):
        super(Xyz2LocalSDT, self).__init__()
        self.name = "xyz2local_sdt"
        self.parent = parent

    def __repr__(self):
        return f'xyz2local_sdt(ns_xyz[3], spacing=(D,H,W), scale=1.) -> ns_xyz[D,H,W,3], ns_local_sdt[D,H,W,1]'

    def __call__(self, ns_xyz: (float, float, float), spacing=(2, 2, 2), scale=1.) -> (jnp.ndarray, jnp.ndarray):
        ns_xyz = jnp.array(ns_xyz)
        ns_cube = grid_in_cube(spacing=spacing, scale=scale, center_shift=(0., 0., 0.))
        ns_cube = ns_cube + ns_xyz
        ns_cube = ns_cube.reshape((-1, 3))
        ns_local_sdt = self.parent.ns_xyz2sdt(ns_cube)
        ns_local_sdt = ns_local_sdt.reshape(spacing)[:, :, :, jnp.newaxis]
        ns_cube = ns_cube.reshape((spacing[0], spacing[1], spacing[2], 3))
        return ns_cube, ns_local_sdt
