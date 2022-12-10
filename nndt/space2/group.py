from colorama import Fore

from nndt.space2.abstracts import AbstractBBoxNode, IterAccessMixin


class Group(AbstractBBoxNode, IterAccessMixin):
    """
    Group is a tree element that can contain other groups and 3D objects.

    :param name: name of the tree node
    :type name: str
    :param bbox: boundary box in form ((X_min, Y_min, Z_min), (X_max, Y_max, Z_max)),
        defaults to ((0., 0., 0.), (0., 0., 0.)).
    :type bbox: tuple, optional
    :param parent: parent node, defaults to None.
    :type parent: class: `Space`, optional
    """

    def __init__(self, name, bbox=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), parent=None):
        super(Group, self).__init__(
            name, parent=parent, bbox=bbox, _print_color=Fore.RED, _nodetype="G"
        )
