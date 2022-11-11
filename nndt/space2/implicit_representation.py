import jax
from colorama import Fore

from nndt.primitive_sdf import AbstractSDF
from nndt.space2.abstracts import AbstractBBoxNode, IterAccessMixin, node_method
from nndt.trainable_task import SimpleSDF


class ImpRepr(AbstractBBoxNode, IterAccessMixin):
    """
    Specifies an implicit representation.


    Args:
        name (str): Name.
        abstract_sdf (AbstractSDF): Abstract signed distance function.
        parent (_type_, optional): Parent node. Defaults to None.
    """

    def __init__(self, name: str, abstract_sdf: AbstractSDF, parent=None):
        super(ImpRepr, self).__init__(
            name,
            parent=parent,
            bbox=abstract_sdf.bbox,
            _print_color=Fore.MAGENTA,
            _nodetype="IR",
        )

        self.abstract_sdf = abstract_sdf

        # TODO This is a place for improvement
        # I name this variable _loader, for compatibility with SDTMethodSetNode
        self._loader = abstract_sdf

    @node_method("purefun_sdf()")
    def purefun_sdf(self):
        """Return signed distance function."""
        return self.abstract_sdf.fun

    @node_method("purefun_vec_sdf()")
    def purefun_vec_sdf(self):
        """Return signed distance function in vector view"""
        return self.abstract_sdf.vec_fun

    @node_method("purefun_vec_sdf_dx()")
    def purefun_vec_sdf_dx(self):
        """Return signed distance function in vector view along the X-axis"""
        return self.abstract_sdf.vec_fun_dx

    @node_method("purefun_vec_sdf_dy()")
    def purefun_vec_sdf_dy(self):
        """Return signed distance function in vector view along the Y-axis"""
        return self.abstract_sdf.vec_fun_dy

    @node_method("purefun_vec_sdf_dz()")
    def purefun_vec_sdf_dz(self):
        """Return signed distance function in vector view along the Z-axis"""
        return self.abstract_sdf.vec_fun_dz


class IR1SDF(AbstractSDF):
    def __init__(self, func: SimpleSDF.FUNC, params, bbox):
        self.func = func
        self.params = params
        self._bbox = bbox

        key = jax.random.PRNGKey(42)
        self._fun = lambda x, y, z: self.func.sdf(params, key, x, y, z)
        self._vec_fun = lambda x, y, z: self.func.vec_sdf(params, key, x, y, z)
        self._vec_fun_x = lambda x, y, z: self.func.vec_sdf_dx(params, key, x, y, z)
        self._vec_fun_y = lambda x, y, z: self.func.vec_sdf_dy(params, key, x, y, z)
        self._vec_fun_z = lambda x, y, z: self.func.vec_sdf_dz(params, key, x, y, z)

    @property
    def bbox(self) -> ((float, float, float), (float, float, float)):
        return self._bbox

    def _get_fun(self):
        return self._fun
