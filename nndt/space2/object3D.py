from colorama import Fore

from nndt.space2.abstracts import AbstractBBoxNode, IterAccessMixin


class Object3D(AbstractBBoxNode, IterAccessMixin):
    """
    An object that can only contain data sources. Сan be represented by multiple files.
    """
    
    def __init__(self, name,
                 bbox=((0., 0., 0.), (0., 0., 0.)),
                 parent=None):
        """
        Args:
            name (str): Name of object.
            bbox (tuple, optional):boundary box in form ((X_min, Y_min, Z_min), (X_max, Y_max, Z_max)).
            parent (_type_, optional): parent node. Defaults to None.
        """
        super(Object3D, self).__init__(name, parent=parent, bbox=bbox, _print_color=Fore.BLUE, _nodetype='O3D')

    def __repr__(self):
        return self._print_color + f'{self._nodetype}:{self.name}' + Fore.WHITE + f' {self._print_bbox()}' + Fore.RESET
