from colorama import Fore

from nndt.primitive_sdf import AbstractSDF
from nndt.space2.abstracts import AbstractBBoxNode, IterAccessMixin, node_method


class ImpRepr(AbstractBBoxNode, IterAccessMixin):
    def __init__(self, name: str, abstract_sdf: AbstractSDF, parent=None):
        super(ImpRepr, self).__init__(
            name,
            parent=parent,
            bbox=abstract_sdf.bbox,
            _print_color=Fore.MAGENTA,
            _nodetype="IR",
        )

        self.abstract_sdf = abstract_sdf

        # This is a place for improvement
        # I name this variable _loader, for compatibility with SDTMethodSetNode
        self._loader = abstract_sdf

    @node_method("purefun_sdf()")
    def purefun_sdf(self):
        return self.abstract_sdf.fun

    @node_method("purefun_vec_sdf()")
    def purefun_vec_sdf(self):
        return self.abstract_sdf.vec_fun

    @node_method("purefun_vec_sdf_dx()")
    def purefun_vec_sdf_dx(self):
        return self.abstract_sdf.vec_fun_dx

    @node_method("purefun_vec_sdf_dy()")
    def purefun_vec_sdf_dy(self):
        return self.abstract_sdf.vec_fun_dy

    @node_method("purefun_vec_sdf_dz()")
    def purefun_vec_sdf_dz(self):
        return self.abstract_sdf.vec_fun_dz
