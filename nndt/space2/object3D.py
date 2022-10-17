from colorama import Fore

from nndt.space2.abstracts import AbstractBBoxNode, IterAccessMixin


class Object3D(AbstractBBoxNode, IterAccessMixin):
    def __init__(self, name,
                 bbox=((0., 0., 0.), (0., 0., 0.)),
                 parent=None):
        super(Object3D, self).__init__(name, parent=parent, bbox=bbox, _print_color=Fore.BLUE, _nodetype='O3D')

    def __repr__(self):
        return self._print_color + f'{self._nodetype}:{self.name}' + Fore.WHITE + f' {self._print_bbox()}' + Fore.RESET
