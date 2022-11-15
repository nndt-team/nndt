import re
from abc import abstractmethod
from functools import wraps
from typing import Optional, Union

from anytree import NodeMixin, PostOrderIter, Resolver
from colorama import Fore
from nndt.vizualize import ANSIConverter

FORBIDDEN_NAME = [
    "separator",
    "parent",
    "__check_loop",
    "__detach",
    "__attach",
    "__children_or_empty",
    "children",
    "__check_children",
    "children_setter",
    "children_deleter",
    "_pre_detach_children",
    "_post_detach_children",
    "_pre_attach_children",
    "_post_attach_children",
    "path",
    "iter_path_reverse",
    "_path",
    "ancestors",
    "anchestors",
    "descendants",
    "root",
    "siblings",
    "leaves",
    "is_leaf",
    "is_root",
    "height",
    "depth",
    "_pre_detach",
    "_post_detach",
    "_pre_attach",
    "_post_attach",
] + ["name", "parent", "_print_color", "_nodetype"]

NODE_METHOD_DICT = {}

DICT_NODETYPE_PRIORITY = {
    "S": 100,
    "G": 90,
    "O3D": 80,
    "IR": 70,
    "FS": 60,
    "TR": 50,
    "MS": 40,
    "M": 30,
}


def node_method(helpstr=None):
    """
    This decorator mark method for attachments to the space model tree

    :param helpstr: Short information about a performed transformation of data
    :return: wrapper
    """

    def decorator_wrapper(fn):
        classname = str(fn.__qualname__).split(".")[0]
        if classname not in NODE_METHOD_DICT:
            NODE_METHOD_DICT[classname] = {}
        NODE_METHOD_DICT[classname][str(fn.__name__)] = helpstr

        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    return decorator_wrapper


def _name_to_safename(name: str) -> str:
    safe_name = re.sub("\W|^(?=\d)", "_", name)
    if safe_name in FORBIDDEN_NAME:
        safe_name = safe_name + "_"
        if safe_name in FORBIDDEN_NAME:
            safe_name = safe_name + "_"
            if safe_name in FORBIDDEN_NAME:
                raise ValueError(f"{name} cannot be safely renamed.")
    return safe_name


class AbstractTreeElement(NodeMixin):
    """
    Abstract element of the space model tree
    """

    def __init__(
        self,
        name: str,
        _print_color: str = Fore.RESET,
        _nodetype: str = "UNDEFINED",
        parent=None,
    ):
        """
        Create an abstract element of space model tree

        :param name: name of the tree node
        :param _print_color: color of this node for print()
        :param _nodetype: type of the code in form of string literal
        :param parent: parent node
        """
        super(NodeMixin, self).__init__()
        self.name = _name_to_safename(name)
        self.parent = parent
        self._print_color = _print_color
        self._nodetype = _nodetype

        self._resolver = Resolver("name")

    def _container_only_list(self):
        return [ch for ch in self.children if isinstance(ch, IterAccessMixin)]

    def __len__(self):
        return len(self._container_only_list())

    def __iter__(self):
        return iter(self._container_only_list())

    def __getitem__(self, request_: Union[int, str]):
        if isinstance(request_, int):
            children_without_methods = self._container_only_list()
            return children_without_methods[request_]
        elif isinstance(request_, str):
            return self._resolver.get(self, request_)
        else:
            raise NotImplementedError()

    def __repr__(self):
        return self._print_color + f"{self._nodetype}:{self.name}" + Fore.RESET

    def _repr_html_(self):
        return (
            '<p>'
            + f'<span style=\"color:{ANSIConverter(self._print_color, type="Fore").to_rgb()}\">'
            + f'{self._nodetype}:{self.name}'
            + '</span>'
            + '</p>'
        )

    def _post_attach(self, parent):
        if isinstance(self, AbstractBBoxNode):
            if parent is not None:
                setattr(parent, self.name, self)

    def _post_detach(self, parent):
        if parent is not None:
            if hasattr(parent, self.name):
                delattr(parent, self.name)


class AbstractBBoxNode(AbstractTreeElement):
    """
    Element of the space model tree with a boundary box
    """

    def __init__(
        self,
        name: str,
        bbox=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        _print_color=Fore.RESET,
        _nodetype="UNDEFINED",
        parent=None,
    ):
        """
        Create an element of the space model tree with a boundary box

        :param name: name of the tree node
        :param bbox: boundary box in form ((X_min, Y_min, Z_min), (X_max, Y_max, Z_max))
        :param parent: parent node
        """
        super(AbstractBBoxNode, self).__init__(
            name, _print_color=_print_color, _nodetype=_nodetype, parent=parent
        )
        self.bbox = bbox

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

    def _print_bbox(self):
        a = self.bbox
        return f"(({a[0][0]:.02f}, {a[0][1]:.02f}, {a[0][2]:.02f}), ({a[1][0]:.02f}, {a[1][1]:.02f}, {a[1][2]:.02f}))"

    @node_method("print(default|source|full)")
    def print(self, mode: Optional[str] = "default"):
        """
        Print tree view for this element and all children's elements

        Args:
            mode (Optional[str], optional):
                "default" print 3D objects and methods;
                "source" prints only space, groups, and file sources;
                "full" prints all nodes of the tree;
                Defaults to "default".

        Returns:
            str: tree representation string
        """
        from nndt.space2.print_tree import _pretty_print

        return _pretty_print(self, mode)

    @node_method("plot(default, filepath=None)")
    def plot(
        self, mode: Optional[str] = "default", filepath: Optional[str] = None, **kwargs
    ):
        """
        Iterate over all Object3D in the tree and show the plot if the filepath is None.
        Save the plot in a file if the filepath is None.

        Args:
            mode (Optional[str], optional):
                Only the "default" mode is supported yet. Defaults to "default".
            filepath (Optional[str], optional):
                File name. If it exists save there otherwise show it. Defaults to None.
        """
        from nndt.space2.plot_tree import _plot

        _plot(self, mode, filepath, **kwargs)

    @node_method("unload_from_memory()")
    def unload_from_memory(self):
        from nndt.space2.filesource import FileSource

        for node in PostOrderIter(self):
            if isinstance(node, FileSource) and node._loader is not None:
                node._loader.unload_data()


class IterAccessMixin:
    pass


class AbstractLoader:
    def calc_bbox(self) -> ((float, float, float), (float, float, float)):
        """Return the boundary box size of a 3D object.

        Returns:
            (tuple), (tuple): boundary box: (Xmin, Xmax, Ymin), (Ymax, Zmin, Zmax)
        """
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    @abstractmethod
    def load_data(self):
        """Load data to memory"""
        pass

    @abstractmethod
    def unload_data(self):
        """Remove references to the uploaded data. This method NO NOT call the garbage collectors"""
        pass

    @abstractmethod
    def is_load(self) -> bool:
        """Check if the data is already loaded"""
        pass
