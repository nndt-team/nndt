from colorama import Fore

from nndt.space2.utils import update_bbox
from nndt.space2.abstracts import AbstractBBoxNode


class Group(AbstractBBoxNode):
    """
    Group is an element tree that is a container for other groups and 3D objects
    """

    def __init__(self, name,
                 bbox=((0., 0., 0.), (0., 0., 0.)),
                 parent=None):
        """
        Create a group. Group is an element tree that is a container for other groups and 3D objects

        :param name: name of the tree node
        :param bbox: boundary box in form ((X_min, Y_min, Z_min), (X_max, Y_max, Z_max))
        :param parent: parent node
        """
        super(Group, self).__init__(name, parent=parent, bbox=bbox, _print_color=Fore.RED, _nodetype='G')

    # def _initialization(self, mode='ident', scale=50, keep_in_memory=False):
    #     for child in self.children:
    #         if isinstance(child, AbstractBBoxNode):
    #             self.bbox = update_bbox(self.bbox, child.bbox)
    #
    #     from nndt.space2 import SamplingNode
    #     SamplingNode(parent=self)
