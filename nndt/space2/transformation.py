from abc import abstractmethod
from typing import *

import jax.numpy as jnp
import numpy as onp
from colorama import Fore

from nndt.space2 import AbstractBBoxNode, node_method


class AbstractTransformation(AbstractBBoxNode):

    def __init__(self, name: str,
                 bbox=((0., 0., 0.), (0., 0., 0.)),
                 parent=None):
        super(AbstractTransformation, self).__init__(name,
                                                     bbox=bbox, _print_color=Fore.RED, _nodetype='TR',
                                                     parent=parent)

        self._transform_type = "unknown_transform"

    def __len__(self):
        return len(self.children)

    def __getitem__(self, request_: Union[int, str]):

        if isinstance(request_, int):
            children_without_methods = [ch for ch in self.children if isinstance(ch, AbstractBBoxNode)]
            return children_without_methods[request_]
        elif isinstance(request_, str):
            return self.resolver.get(self, request_)
        else:
            raise NotImplementedError()

    def __repr__(self):
        return self._print_color + f'{self._nodetype}:{self.name}' + Fore.WHITE + f" {self._transform_type}" + Fore.RESET

    def _print_bbox(self):
        a = self.bbox
        return f"(({a[0][0]:.02f}, {a[0][1]:.02f}, {a[0][2]:.02f}), ({a[1][0]:.02f}, {a[1][1]:.02f}, {a[1][2]:.02f}))"

    def _post_attach(self, parent):
        if parent is not None:
            setattr(parent, self.name, self)

    def _post_detach(self, parent):
        if parent is not None:
            if hasattr(parent, self.name):
                delattr(parent, self.name)

    @abstractmethod
    def transform_xyz_ps2ns(self, xyz: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        pass

    @abstractmethod
    def transform_xyz_ns2ps(self, xyz: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        pass

    @abstractmethod
    def transform_sdt_ns2ps(self, sdt: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        pass

    @abstractmethod
    def transform_sdt_ps2ns(self, sdt: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        pass


class IdentityTransform(AbstractTransformation):

    def __init__(self, ps_bbox: ((float, float, float), (float, float, float)),
                 parent=None):
        super(IdentityTransform, self).__init__("transform", bbox=ps_bbox, parent=parent)

        self.bbox = ps_bbox
        self._transform_type = "identity"

    @node_method("transform_xyz_ps2ns(ps_xyz[...,3]) -> ns_xyz[...,3]")
    def transform_xyz_ps2ns(self, xyz: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        """Transforms xyz physical space to normalized space

        Args:
            xyz (Union[onp.ndarray, jnp.ndarray]): xyz in physical space.

        Returns:
            Union[onp.ndarray, jnp.ndarray]: xyz in normalized space.
        """
        return xyz

    @node_method("transform_xyz_ns2ps(ns_xyz[...,3]) -> ps_xyz[...,3]")
    def transform_xyz_ns2ps(self, xyz: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        """Transforms xyz normalized space to physical space

        Args:
            xyz (Union[onp.ndarray, jnp.ndarray]): xyz in normalized space.

        Returns:
            Union[onp.ndarray, jnp.ndarray]: xyz in physical space.
        """
        return xyz

    @node_method("transform_sdt_ns2ps(ns_sdt[...]) -> ps_sdt[...]")
    def transform_sdt_ns2ps(self, sdt: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        """Transforms signed distance tensor of normalized space to physical space.
        
        Args:
            sdt (Union[onp.ndarray, jnp.ndarray]): Signed distance tensor in normalized space.

        Returns:
            Union[onp.ndarray, jnp.ndarray]: Signed distance tensor in physical space.
        """
        return sdt

    @node_method("transform_sdt_ps2ns(ps_sdt[...]) -> ns_sdt[...]")
    def transform_sdt_ps2ns(self, sdt: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        """Transforms signed distance tensor of physical space to normalized space.
        
        Args:
            sdt (Union[onp.ndarray, jnp.ndarray]): Signed distance tensor in physical space.

        Returns:
            Union[onp.ndarray, jnp.ndarray]:  Signed distance tensor in normalized space.
        """
        return sdt


class ShiftAndScaleTransform(AbstractTransformation):

    def __init__(self, ps_bbox: ((float, float, float), (float, float, float)),
                 ps_center: (float, float, float),
                 ns_center: (float, float, float),
                 scale_ps2ns: float,
                 parent=None):
        super(ShiftAndScaleTransform, self).__init__("transform", parent=parent)

        self.ps_center = jnp.array(ps_center)
        self.ns_center = jnp.array(ns_center)
        self.scale_ps2ns = scale_ps2ns
        self._transform_type = "shift_and_scale"

        bbox_ = self.transform_xyz_ps2ns(jnp.array(ps_bbox))
        self.bbox = ((float(bbox_[0][0]), float(bbox_[0][1]), float(bbox_[0][2])),
                     (float(bbox_[1][0]), float(bbox_[1][1]), float(bbox_[1][2])))

    @node_method("transform_xyz_ps2ns(ps_xyz[...,3]) -> ns_xyz[...,3]")
    def transform_xyz_ps2ns(self, xyz: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        """Transforms xyz physical space to normalized space

        Args:
            xyz (Union[onp.ndarray, jnp.ndarray]): xyz in physical space.

        Returns:
            Union[onp.ndarray, jnp.ndarray]: xyz in normalized space.
        """
        return (xyz - jnp.array(self.ps_center)) / self.scale_ps2ns + jnp.array(self.ns_center)

    @node_method("transform_xyz_ns2ps(ns_xyz[...,3]) -> ps_xyz[...,3]")
    def transform_xyz_ns2ps(self, xyz: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        """Transforms xyz normalized space to physical space

        Args:
            xyz (Union[onp.ndarray, jnp.ndarray]): xyz in normalized space.

        Returns:
            Union[onp.ndarray, jnp.ndarray]: xyz in physical space.
        """
        return (xyz - jnp.array(self.ns_center)) * self.scale_ps2ns + jnp.array(self.ps_center)

    @node_method("transform_sdt_ns2ps(ns_sdt[...]) -> ps_sdt[...]")
    def transform_sdt_ns2ps(self, sdt: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        """Transforms signed distance tensor of normalized space to physical space.
        
        Args:
            sdt (Union[onp.ndarray, jnp.ndarray]): Signed distance tensor in normalized space.

        Returns:
            Union[onp.ndarray, jnp.ndarray]: Signed distance tensor in physical space.
        """
        return sdt * self.scale_ps2ns

    @node_method("transform_sdt_ps2ns(ps_sdt[...]) -> ns_sdt[...]")
    def transform_sdt_ps2ns(self, sdt: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        """Transforms signed distance tensor of physical space to normalized space.
        
        Args:
            sdt (Union[onp.ndarray, jnp.ndarray]): Signed distance tensor in physical space.

        Returns:
            Union[onp.ndarray, jnp.ndarray]:  Signed distance tensor in normalized space.
        """
        return sdt / self.scale_ps2ns


class ToNormalCubeTransform(AbstractTransformation):

    def __init__(self, ps_bbox: ((float, float, float), (float, float, float)),
                 parent=None):
        super(ToNormalCubeTransform, self).__init__("transform", parent=parent)

        self.ps_lower = jnp.array(ps_bbox[0])
        self.ps_upper = jnp.array(ps_bbox[1])
        self.ps_center = (self.ps_lower + self.ps_upper) / 2.
        self.scale = (self.ps_upper - self.ps_lower) / 2.

        self.bbox = ((-1., -1., -1.), (1., 1., 1.))

        self._transform_type = "to_cube"

    @node_method("transform_xyz_ps2ns(ps_xyz[...,3]) -> ns_xyz[...,3]")
    def transform_xyz_ps2ns(self, xyz: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        """Transforms xyz physical space to normalized space

        Args:
            xyz (Union[onp.ndarray, jnp.ndarray]): xyz in physical space.

        Returns:
            Union[onp.ndarray, jnp.ndarray]: xyz in normalized space.
        """
        return (xyz - self.ps_center) / self.scale

    @node_method("transform_xyz_ns2ps(ns_xyz[...,3]) -> ps_xyz[...,3]")
    def transform_xyz_ns2ps(self, xyz: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        """Transforms xyz normalized space to physical space

        Args:
            xyz (Union[onp.ndarray, jnp.ndarray]): xyz in normalized space.

        Returns:
            Union[onp.ndarray, jnp.ndarray]: xyz in physical space.
        """
        return (xyz * self.scale) + self.ps_center

    @node_method("transform_sdt_ns2ps(ns_sdt[...]) -> ps_sdt[...]")
    def transform_sdt_ns2ps(self, sdt: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        """Transforms signed distance tensor of normalized space to physical space.
        
        Args:
            sdt (Union[onp.ndarray, jnp.ndarray]): Signed distance tensor in normalized space.

        Returns:
            Union[onp.ndarray, jnp.ndarray]: Signed distance tensor in physical space.
        """
        return sdt * self.scale

    @node_method("transform_sdt_ps2ns(ps_sdt[...]) -> ns_sdt[...]")
    def transform_sdt_ps2ns(self, sdt: Union[onp.ndarray, jnp.ndarray]) -> Union[onp.ndarray, jnp.ndarray]:
        """Transforms signed distance tensor of physical space to normalized space.
        
        Args:
            sdt (Union[onp.ndarray, jnp.ndarray]): Signed distance tensor in physical space.

        Returns:
            Union[onp.ndarray, jnp.ndarray]:  Signed distance tensor in normalized space.
        """
        return sdt / self.scale
