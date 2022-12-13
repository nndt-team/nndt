from abc import abstractmethod
from typing import *

import jax.numpy as jnp
import numpy as onp
from colorama import Fore
from nndt.vizualize import ANSIConverter

from nndt.space2.abstracts import AbstractBBoxNode, node_method


class AbstractTransformation(AbstractBBoxNode):
    def __init__(self, name: str, bbox=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), parent=None):
        super(AbstractTransformation, self).__init__(
            name, bbox=bbox, _print_color=Fore.RED, _nodetype="TR", parent=parent
        )

        self._transform_type = "unknown_transform"

    def __len__(self):
        return len(self.children)

    def __getitem__(self, request_: Union[int, str]):

        if isinstance(request_, int):
            children_without_methods = [
                ch for ch in self.children if isinstance(ch, AbstractBBoxNode)
            ]
            return children_without_methods[request_]
        elif isinstance(request_, str):
            return self.resolver.get(self, request_)
        else:
            raise NotImplementedError()

    def __repr__(self):
        return (
            self._print_color
            + f"{self._nodetype}:{self.name}"
            + Fore.WHITE
            + f" {self._transform_type}"
            + Fore.RESET
        )

    def _repr_html_(self):
        return (
            '<p>'
            + f'<span style=\"color:{ANSIConverter(self._print_color, type="Fore").to_rgb()}\">'
            + f'{self._nodetype}:{self.name}'
            + '</span>'
            + f'<span style=\"color:{ANSIConverter(Fore.WHITE, type="Fore").to_rgb()}\">'
            + f' {self._transform_type}'
            + '</span>'
            + '</p>'
        )

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
    def transform_xyz_ps2ns(
        self, xyz: Union[onp.ndarray, jnp.ndarray]
    ) -> Union[onp.ndarray, jnp.ndarray]:
        pass

    @abstractmethod
    def transform_xyz_ns2ps(
        self, xyz: Union[onp.ndarray, jnp.ndarray]
    ) -> Union[onp.ndarray, jnp.ndarray]:
        pass

    @abstractmethod
    def transform_sdt_ns2ps(
        self, sdt: Union[onp.ndarray, jnp.ndarray]
    ) -> Union[onp.ndarray, jnp.ndarray]:
        pass

    @abstractmethod
    def transform_sdt_ps2ns(
        self, sdt: Union[onp.ndarray, jnp.ndarray]
    ) -> Union[onp.ndarray, jnp.ndarray]:
        pass


class IdentityTransform(AbstractTransformation):
    """
    Transfer object of the physical space to normalized space without any changes.

    Args:
        ps_bbox (tuple, optional): boundary box in form ((X_min, Y_min, Z_min), (X_max, Y_max, Z_max)).
        parent (_type_, optional): parent node. Defaults to None.
    """

    def __init__(
        self, ps_bbox: ((float, float, float), (float, float, float)), parent=None
    ):
        super(IdentityTransform, self).__init__(
            "transform", bbox=ps_bbox, parent=parent
        )

        self.bbox = ps_bbox
        self._transform_type = "identity"

    @node_method("transform_xyz_ps2ns(ps_xyz[..,3]) -> ns_xyz[..,3]")
    def transform_xyz_ps2ns(
        self, xyz: Union[onp.ndarray, jnp.ndarray]
    ) -> Union[onp.ndarray, jnp.ndarray]:
        """Transforms xyz physical space to normalized space

        Args:
            xyz (Union[onp.ndarray, jnp.ndarray]): xyz in physical space.

        Returns:
            Union[onp.ndarray, jnp.ndarray]: xyz in normalized space.
        """
        return xyz

    @node_method("transform_xyz_ns2ps(ns_xyz[..,3]) -> ps_xyz[..,3]")
    def transform_xyz_ns2ps(
        self, xyz: Union[onp.ndarray, jnp.ndarray]
    ) -> Union[onp.ndarray, jnp.ndarray]:
        """Transforms xyz normalized space to physical space

        Args:
            xyz (Union[onp.ndarray, jnp.ndarray]): xyz in normalized space.

        Returns:
            Union[onp.ndarray, jnp.ndarray]: xyz in physical space.
        """
        return xyz

    @node_method("transform_sdt_ns2ps(ns_sdt[..]) -> ps_sdt[..]")
    def transform_sdt_ns2ps(
        self, sdt: Union[onp.ndarray, jnp.ndarray]
    ) -> Union[onp.ndarray, jnp.ndarray]:
        """Transforms signed distance tensor of normalized space to physical space.

        Args:
            sdt (Union[onp.ndarray, jnp.ndarray]): Signed distance tensor in normalized space.

        Returns:
            Union[onp.ndarray, jnp.ndarray]: Signed distance tensor in physical space.
        """
        return sdt

    @node_method("transform_sdt_ps2ns(ps_sdt[..]) -> ns_sdt[..]")
    def transform_sdt_ps2ns(
        self, sdt: Union[onp.ndarray, jnp.ndarray]
    ) -> Union[onp.ndarray, jnp.ndarray]:
        """Transforms signed distance tensor of physical space to normalized space.

        Args:
            sdt (Union[onp.ndarray, jnp.ndarray]): Signed distance tensor in physical space.

        Returns:
            Union[onp.ndarray, jnp.ndarray]:  Signed distance tensor in normalized space.
        """
        return sdt


class ShiftAndScaleTransform(AbstractTransformation):
    """
    Transfer objects from a physical space to a normalized space using shift and scale transformation.

    Args:
        ps_bbox (float, float, float): boundary box in a form.
        ps_center (float, float, float): Physical space center.
        ns_center (float, float, float): Normalized space center.
        scale_ps2ns (float): Scale.
        parent (_type_, optional): parent node. Defaults to None.
    """

    def __init__(
        self,
        ps_bbox: ((float, float, float), (float, float, float)),
        ps_center: (float, float, float),
        ns_center: (float, float, float),
        scale_ps2ns: float,
        parent=None,
    ):
        super(ShiftAndScaleTransform, self).__init__("transform", parent=parent)

        self.ps_center = jnp.array(ps_center)
        self.ns_center = jnp.array(ns_center)
        self.scale_ps2ns = scale_ps2ns
        self._transform_type = "shift_and_scale"

        bbox_ = self.transform_xyz_ps2ns(jnp.array(ps_bbox))
        self.bbox = (
            (float(bbox_[0][0]), float(bbox_[0][1]), float(bbox_[0][2])),
            (float(bbox_[1][0]), float(bbox_[1][1]), float(bbox_[1][2])),
        )

    @node_method("transform_xyz_ps2ns(ps_xyz[..,3]) -> ns_xyz[..,3]")
    def transform_xyz_ps2ns(
        self, xyz: Union[onp.ndarray, jnp.ndarray]
    ) -> Union[onp.ndarray, jnp.ndarray]:
        """Transforms xyz physical space to normalized space

        Args:
            xyz (Union[onp.ndarray, jnp.ndarray]): xyz in physical space.

        Returns:
            Union[onp.ndarray, jnp.ndarray]: xyz in normalized space.
        """
        return (xyz - jnp.array(self.ps_center)) / self.scale_ps2ns + jnp.array(
            self.ns_center
        )

    @node_method("transform_xyz_ns2ps(ns_xyz[..,3]) -> ps_xyz[..,3]")
    def transform_xyz_ns2ps(
        self, xyz: Union[onp.ndarray, jnp.ndarray]
    ) -> Union[onp.ndarray, jnp.ndarray]:
        """Transforms xyz normalized space to physical space

        Args:
            xyz (Union[onp.ndarray, jnp.ndarray]): xyz in normalized space.

        Returns:
            Union[onp.ndarray, jnp.ndarray]: xyz in physical space.
        """
        return (xyz - jnp.array(self.ns_center)) * self.scale_ps2ns + jnp.array(
            self.ps_center
        )

    @node_method("transform_sdt_ns2ps(ns_sdt[..]) -> ps_sdt[..]")
    def transform_sdt_ns2ps(
        self, sdt: Union[onp.ndarray, jnp.ndarray]
    ) -> Union[onp.ndarray, jnp.ndarray]:
        """Transforms signed distance tensor of normalized space to physical space.

        Args:
            sdt (Union[onp.ndarray, jnp.ndarray]): Signed distance tensor in normalized space.

        Returns:
            Union[onp.ndarray, jnp.ndarray]: Signed distance tensor in physical space.
        """
        return sdt * self.scale_ps2ns

    @node_method("transform_sdt_ps2ns(ps_sdt[..]) -> ns_sdt[..]")
    def transform_sdt_ps2ns(
        self, sdt: Union[onp.ndarray, jnp.ndarray]
    ) -> Union[onp.ndarray, jnp.ndarray]:
        """Transforms signed distance tensor of physical space to normalized space.

        Args:
            sdt (Union[onp.ndarray, jnp.ndarray]): Signed distance tensor in physical space.

        Returns:
            Union[onp.ndarray, jnp.ndarray]:  Signed distance tensor in normalized space.
        """
        return sdt / self.scale_ps2ns


class ToNormalCubeTransform(AbstractTransformation):
    """
    Transfer objects from a physical space to a normalized space.
    All objects are scaled to cubes with coordinates from -1 to 1.
    This approach is similar to NormalScaler in scikit-learn.

    Args:
        ps_bbox (float, float, float):  boundary box in the form.
        parent (_type_, optional):  parent node. Defaults to None.
    """

    def __init__(
        self, ps_bbox: ((float, float, float), (float, float, float)), parent=None
    ):
        super(ToNormalCubeTransform, self).__init__("transform", parent=parent)

        self.ps_lower = jnp.array(ps_bbox[0])
        self.ps_upper = jnp.array(ps_bbox[1])
        self.ps_center = (self.ps_lower + self.ps_upper) / 2.0
        self.scale = (self.ps_upper - self.ps_lower) / 2.0

        self.bbox = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))

        self._transform_type = "to_cube"

    @node_method("transform_xyz_ps2ns(ps_xyz[..,3]) -> ns_xyz[..,3]")
    def transform_xyz_ps2ns(
        self, xyz: Union[onp.ndarray, jnp.ndarray]
    ) -> Union[onp.ndarray, jnp.ndarray]:
        """Transforms xyz physical space to normalized space

        Args:
            xyz (Union[onp.ndarray, jnp.ndarray]): xyz in physical space.

        Returns:
            Union[onp.ndarray, jnp.ndarray]: xyz in normalized space.
        """
        return (xyz - self.ps_center) / self.scale

    @node_method("transform_xyz_ns2ps(ns_xyz[..,3]) -> ps_xyz[..,3]")
    def transform_xyz_ns2ps(
        self, xyz: Union[onp.ndarray, jnp.ndarray]
    ) -> Union[onp.ndarray, jnp.ndarray]:
        """Transforms xyz normalized space to physical space

        Args:
            xyz (Union[onp.ndarray, jnp.ndarray]): xyz in normalized space.

        Returns:
            Union[onp.ndarray, jnp.ndarray]: xyz in physical space.
        """
        return (xyz * self.scale) + self.ps_center

    @node_method("transform_sdt_ns2ps(ns_sdt[..]) -> ps_sdt[..]")
    def transform_sdt_ns2ps(
        self, sdt: Union[onp.ndarray, jnp.ndarray]
    ) -> Union[onp.ndarray, jnp.ndarray]:
        """Transforms signed distance tensor of normalized space to physical space.

        Args:
            sdt (Union[onp.ndarray, jnp.ndarray]): Signed distance tensor in normalized space.

        Returns:
            Union[onp.ndarray, jnp.ndarray]: Signed distance tensor in physical space.
        """
        return sdt * self.scale

    @node_method("transform_sdt_ps2ns(ps_sdt[..]) -> ns_sdt[..]")
    def transform_sdt_ps2ns(
        self, sdt: Union[onp.ndarray, jnp.ndarray]
    ) -> Union[onp.ndarray, jnp.ndarray]:
        """Transforms signed distance tensor of physical space to normalized space.

        Args:
            sdt (Union[onp.ndarray, jnp.ndarray]): Signed distance tensor in physical space.

        Returns:
            Union[onp.ndarray, jnp.ndarray]:  Signed distance tensor in normalized space.
        """
        return sdt / self.scale
