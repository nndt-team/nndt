from colorama import Fore

from nndt.primitive_sdf import AbstractSDF
from nndt.space2.abstracts import AbstractBBoxNode, IterAccessMixin


class ImpRepr(AbstractBBoxNode, IterAccessMixin):

    def __init__(self,
                 name: str,
                 abstract_sdf: AbstractSDF,
                 parent=None):
        super(ImpRepr, self).__init__(name,
                                      parent=parent,
                                      bbox=abstract_sdf.bbox,
                                      _print_color=Fore.MAGENTA,
                                      _nodetype='IR')

        self.abstract_sdf = abstract_sdf

        # This is a place for improvement
        # I name this variable _loader, for compatibility with SDTMethodSetNode
        self._loader = abstract_sdf
