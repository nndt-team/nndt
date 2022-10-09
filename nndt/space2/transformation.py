from abc import abstractmethod
from typing import *

import jax.numpy as jnp
import numpy as onp
from anytree import NodeMixin
from colorama import Fore

from nndt.space2 import AbstractBBoxNode, node_method


class AbstractTransformation(AbstractBBoxNode):

    def __init__(self, name: str,
                 bbox=((0., 0., 0.), (0., 0., 0.)),
                 parent=None):
        super(AbstractTransformation, self).__init__(name,
                 bbox=bbox, _print_color=Fore.RED, _nodetype='T',
                 parent=parent)

        self._transform_type = "unknown_transform"

    def __len__(self):
        return len(self.children)

    def __getitem__(self, request_: Union[int, str]):

        if isinstance(request_, int):
            children_without_methods = [ch for ch in self.children if isinstance(ch, BBoxNode)]
            return children_without_methods[request_]
        elif isinstance(request_, str):
            return self.resolver.get(self, request_)
        else:
            raise NotImplementedError()

    def __repr__(self):
        return self._print_color + f'{self._nodetype}:{self.name}' + Fore.WHITE + f' {self._transform_type}' + Fore.RESET

    def _print_bbox(self):
        a = self.bbox
        return f"(({a[0][0]:.02f}, {a[0][1]:.02f}, {a[0][2]:.02f}), ({a[1][0]:.02f}, {a[1][1]:.02f}, {a[1][2]:.02f}))"

    def _post_attach(self, parent):
        if parent is not None:
            setattr(parent, self.name, self)

        #from nndt.space2 import initialize_method_node
        #initialize_method_node(self)

    def _post_detach(self, parent):
        if parent is not None:
            if hasattr(parent, self.name):
                delattr(parent, self.name)

    def _initialization(self, mode='ident', scale=50, keep_in_memory=False):
        pass

    @abstractmethod
    def xyz_ps2ns(self, xyz: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        pass

    @abstractmethod
    def xyz_ns2ps(self, xyz: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        pass

    @abstractmethod
    def sdt_ns2ps(self, sdt: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        pass

    @abstractmethod
    def sdt_ps2ns(self, sdt: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        pass

class IdentityTransform(AbstractTransformation):

    def __init__(self, ps_bbox: ((float, float, float), (float, float, float)),
                 parent=None):
        super(IdentityTransform, self).__init__("ns", bbox=ps_bbox, parent=parent)

        self.bbox = ps_bbox
        self._transform_type = "ident"

    @node_method("xyz_ps2ns")
    def xyz_ps2ns(self, xyz: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        return xyz

    @node_method("xyz_ns2ps")
    def xyz_ns2ps(self, xyz: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        return xyz

    @node_method("sdt_ns2ps")
    def sdt_ns2ps(self, sdt: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        return sdt

    @node_method("sdt_ps2ns")
    def sdt_ps2ns(self, sdt: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        return sdt

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

    @node_method("xyz_ps2ns")
    def xyz_ps2ns(self, xyz: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        return (xyz - jnp.array(self.ps_center)) / self.scale_ps2ns + jnp.array(self.ns_center)

    @node_method("xyz_ns2ps")
    def xyz_ns2ps(self, xyz: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        return (xyz - jnp.array(self.ns_center)) * self.scale_ps2ns + jnp.array(self.ps_center)

    @node_method("sdt_ns2ps")
    def sdt_ns2ps(self, sdt: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        return sdt * self.scale_ps2ns

    @node_method("sdt_ps2ns")
    def sdt_ps2ns(self, sdt: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        return sdt / self.scale_ps2ns

class ToNormalCubeTransform(AbstractTransformation):

    def __init__(self, ps_bbox: ((float, float, float), (float, float, float)),
                 parent=None):
        super(ToNormalCubeTransform, self).__init__("ns", parent=parent)

        self.ps_lower = jnp.array(ps_bbox[0])
        self.ps_upper = jnp.array(ps_bbox[1])
        self.ps_center = (self.ps_lower + self.ps_upper) / 2.
        self.scale = (self.ps_upper - self.ps_lower) / 2.

        self.bbox = ((-1., -1., -1.), (1., 1., 1.))

        self._transform_type = "to_cube"

    @node_method("xyz_ps2ns")
    def xyz_ps2ns(self, xyz: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        return (xyz - self.ps_center) / self.scale

    @node_method("xyz_ns2ps")
    def xyz_ns2ps(self, xyz: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        return (xyz * self.scale) + self.ps_center

    @node_method("sdt_ns2ps")
    def sdt_ns2ps(self, sdt: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        return sdt * self.scale

    @node_method("sdt_ps2ns")
    def sdt_ps2ns(self, sdt: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        return sdt / self.scale
