from colorama import Fore

from nndt.space2.abstracts import AbstractBBoxNode, IterAccessMixin


class Group(AbstractBBoxNode, IterAccessMixin):
    """
    Group is an element tree that is a container for other groups and 3D objects
    """

    def __init__(self, name,
                 bbox=((0., 0., 0.), (0., 0., 0.)),
                 parent=None):
        """
        :param name: name of the tree node
        :param bbox: boundary box in form ((X_min, Y_min, Z_min), (X_max, Y_max, Z_max))
        :param parent: parent node
        """
        super(Group, self).__init__(name, parent=parent, bbox=bbox, _print_color=Fore.RED, _nodetype='G')
