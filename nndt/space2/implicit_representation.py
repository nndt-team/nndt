from colorama import Fore

from nndt.space2.abstracts import AbstractBBoxNode, IterAccessMixin


class ImpRepr(AbstractBBoxNode, IterAccessMixin):

    def __init__(self, name,
                 bbox=((0., 0., 0.), (0., 0., 0.)),
                 parent=None):
        super(ImpRepr, self).__init__(name, parent=parent, bbox=bbox, _print_color=Fore.MAGENTA, _nodetype='IR')
