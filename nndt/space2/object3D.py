from colorama import Fore
from nndt.vizualize import ANSIConverter

from nndt.space2.abstracts import AbstractBBoxNode, IterAccessMixin


class Object3D(AbstractBBoxNode, IterAccessMixin):
    """
    Object3D is a representation of an object as a whole one. This node can contain data sources, method sets, transformations, and methods.
    Object3D cannot include other objects. One object may be defined by multiple files with various extensions and schemes of data storage.

    Args:
        name (str): Name of the object.
        bbox (tuple, optional): boundary box in form ((X_min, Y_min, Z_min), (X_max, Y_max, Z_max)).
        parent (_type_, optional): parent node. Defaults to None.
    """

    def __init__(self, name, bbox=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), parent=None):
        super(Object3D, self).__init__(
            name, parent=parent, bbox=bbox, _print_color=Fore.GREEN, _nodetype="O3D"
        )

    def __repr__(self):
        return (
            self._print_color
            + f"{self._nodetype}:{self.name}"
            + Fore.WHITE
            + f" {self._print_bbox()}"
            + Fore.RESET
        )

    def _repr_html_(self):
        return (
            '<p>'
            + f'<span style=\"color:{ANSIConverter(self._print_color, type="Fore").to_rgb()}\">'
            + f'{self._nodetype}:{self.name}'
            + '</span>'
            + f'<span style=\"color:{ANSIConverter(Fore.WHITE, type="Fore").to_rgb()}\">'
            + f' {self._print_bbox()}'
            + '</span>'
            + '</p>'
        )

