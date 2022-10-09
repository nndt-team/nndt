import re
from abc import abstractmethod
from typing import Union, Optional

from anytree import NodeMixin, Resolver, RenderTree
from colorama import Fore

FORBIDDEN_NAME = ['separator', 'parent', '__check_loop', '__detach', '__attach', '__children_or_empty', 'children',
                  '__check_children', 'children_setter', 'children_deleter', '_pre_detach_children',
                  '_post_detach_children', '_pre_attach_children', '_post_attach_children', 'path', 'iter_path_reverse',
                  '_path', 'ancestors', 'anchestors', 'descendants', 'root', 'siblings', 'leaves', 'is_leaf', 'is_root',
                  'height', 'depth', '_pre_detach', '_post_detach', '_pre_attach', '_post_attach'] + \
                 ['name', 'parent', '_print_color', '_nodetype']

NODE_METHOD_DICT = {}

DICT_NODETYPE_PRIORITY = {"S": 100, "G": 90, "O3D": 80,
                          "FS": 60, "T": 50, "MS": 40, "M": 30}


def _get_class_hierarchy(obj):
    class_hierarchy = [obj.__class__]
    while len(class_hierarchy[-1].__bases__) > 0:
        class_hierarchy = class_hierarchy + [class_hierarchy[-1].__bases__[0]]
    return class_hierarchy


def node_method(docstring=None):
    def decorator_wrapper(fn):
        classname = str(fn.__qualname__).split('.')[0]
        if classname not in NODE_METHOD_DICT:
            NODE_METHOD_DICT[classname] = {}
        NODE_METHOD_DICT[classname][str(fn.__name__)] = docstring

        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    return decorator_wrapper


def _name_to_safename(name: str) -> str:
    safe_name = re.sub('\W|^(?=\d)', '_', name)
    if safe_name in FORBIDDEN_NAME:
        safe_name = safe_name + '_'
        if safe_name in FORBIDDEN_NAME:
            safe_name = safe_name + '_'
            if safe_name in FORBIDDEN_NAME:
                raise ValueError(f'{name} cannot be safely renamed.')
    return safe_name

#
# def _children_filter_for_explore_default(children):
#     ret = [v for v in children if isinstance(v, AbstractTreeElement)]
#     return ret
#
#
# def _children_filter_for_explore_source(children):
#     ret = [v for v in children if isinstance(v, AbstractBBoxNode)]
#     return ret


class AbstractTreeElement(NodeMixin):
    """
    Abstract element of the space model tree
    """

    def __init__(self, name: str, _print_color: str = Fore.RESET, _nodetype: str = 'UNDEFINED', parent=None):
        """
        Create abstract element of space model tree
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

        self._resolver = Resolver('name')

    def __len__(self):
        return len(self.children)

    def __getitem__(self, request_: Union[int, str]):

        if isinstance(request_, int):
            children_without_methods = [ch for ch in self.children
                                        if isinstance(ch, AbstractBBoxNode)]
            return children_without_methods[request_]
        elif isinstance(request_, str):
            return self._resolver.get(self, request_)
        else:
            raise NotImplementedError()

    def __repr__(self):
        return self._print_color + f'{self._nodetype}:{self.name}' + Fore.RESET

    def _post_attach(self, parent):
        if isinstance(self, AbstractBBoxNode):
            if parent is not None:
                setattr(parent, self.name, self)

    def _post_detach(self, parent):
        if parent is not None:
            if hasattr(parent, self.name):
                delattr(parent, self.name)

    def _add_method_node(self):
        class_hierarchy = _get_class_hierarchy(self)
        class_hierarchy = list([str(class_.__name__) for class_ in class_hierarchy])
        from nndt.space2 import MethodNode
        for class_name in class_hierarchy:
            if class_name in NODE_METHOD_DICT:
                for fn_name, fn_docs in NODE_METHOD_DICT[class_name].items():
                    if hasattr(self, fn_name) and (fn_name not in [x.name for x in self.children]):
                        MethodNode(fn_name, fn_docs, parent=self)


class AbstractBBoxNode(AbstractTreeElement):
    """
    Element of the space model tree with boundary box
    """

    def __init__(self, name: str,
                 bbox=((0., 0., 0.), (0., 0., 0.)),
                 _print_color=Fore.RESET, _nodetype='UNDEFINED',
                 parent=None):
        """
        Create element of the space model tree with boundary box

        :param name: name of the tree node
        :param bbox: boundary box in form ((X_min, Y_min, Z_min), (X_max, Y_max, Z_max))
        :param parent: parent node
        """
        super(AbstractBBoxNode, self).__init__(name, _print_color=_print_color, _nodetype=_nodetype, parent=parent)
        self.bbox = bbox

    def __repr__(self):
        return self._print_color + f'{self._nodetype}:{self.name}' + Fore.WHITE + f' {self._print_bbox()}' + Fore.RESET

    def _print_bbox(self):
        a = self.bbox
        return f"(({a[0][0]:.02f}, {a[0][1]:.02f}, {a[0][2]:.02f}), ({a[1][0]:.02f}, {a[1][1]:.02f}, {a[1][2]:.02f}))"

    @node_method("print(default|source|full)")
    def print(self, mode: Optional[str] = "default"):
        from space2.print_tree import _pretty_print
        return _pretty_print(self, mode)


class AbstractLoader:

    def calc_bbox(self) -> ((float, float, float), (float, float, float)):
        return (0., 0., 0.), (0., 0., 0.)

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def unload_data(self):
        pass

    @abstractmethod
    def is_load(self) -> bool:
        pass
