from abc import abstractmethod
from typing import *

import jax.numpy as jnp
import numpy as onp
from colorama import Fore

from space2 import BBoxNode


class AbstractTransformation(BBoxNode):

    def __init__(self, name: str, parent=None,
                 bbox=((0., 0., 0.), (0., 0., 0.))):
        super(AbstractTransformation, self).__init__(name=name, parent=None,
                                                     bbox=bbox,
                                                     _print_color=Fore.RESET,
                                                     _nodetype='T')

        self.name = name
        self.parent = parent

    @abstractmethod
    def xyz_ps2ns(self, xyz: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        pass

    @abstractmethod
    def xyz_ns2ps(self, xyz: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        pass


class IdentityTransform(AbstractTransformation):

    def __init__(self, ps_bbox: ((float, float, float), (float, float, float)),
                 parent=None):
        super(IdentityTransform, self).__init__("ns", bbox=ps_bbox, parent=parent)

        self.bbox = ps_bbox
        self._transform_type = "ident"

    def xyz_ps2ns(self, xyz: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        return xyz

    def xyz_ns2ps(self, xyz: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        return xyz


class ShiftAndScaleTransform(AbstractTransformation):

    def __init__(self, ps_bbox: ((float, float, float), (float, float, float)),
                 ps_center: (float, float, float),
                 ns_center: (float, float, float),
                 scale_ps2ns: float,
                 parent=None):
        super(ShiftAndScaleTransform, self).__init__("ns", parent=parent)

        self.ps_center = jnp.array(ps_center)
        self.ns_center = jnp.array(ns_center)
        self.scale_ps2ns = scale_ps2ns
        self._transform_type = "shift_and_scale"

        bbox_ = self.xyz_ps2ns(jnp.array(ps_bbox))
        self.bbox = ((float(bbox_[0][0]), float(bbox_[0][1]), float(bbox_[0][2])),
                     (float(bbox_[1][0]), float(bbox_[1][1]), float(bbox_[1][2])))

    def xyz_ps2ns(self, xyz: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        return (xyz - jnp.array(self.ps_center)) / self.scale_ps2ns + jnp.array(self.ns_center)

    def xyz_ns2ps(self, xyz: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        return (xyz - jnp.array(self.ns_center)) * self.scale_ps2ns + jnp.array(self.ps_center)


class ToNormalCubeTransform(AbstractTransformation):

    def __init__(self, ps_bbox: ((float, float, float), (float, float, float)),
                 parent=None):
        super(ToNormalCubeTransform, self).__init__("ns", parent=parent)

        self.ps_lower = jnp.array(ps_bbox[0])
        self.ps_upper = jnp.array(ps_bbox[1])

        self.bbox = ((-1., -1., -1.), (1., 1., 1.))

        self._transform_type = "to_cube"

    def xyz_ps2ns(self, xyz: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        return (2. * (xyz - self.ps_lower) / (self.ps_upper - self.ps_lower)) - 0.5

    def xyz_ns2ps(self, xyz: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        return ((xyz + 0.5) / 2.) * (self.ps_upper - self.ps_lower) + self.ps_lower
