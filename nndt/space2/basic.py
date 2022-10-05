import os
from typing import *

from anytree import NodeMixin, Resolver, RenderTree
from colorama import Fore
import nndt

FORBIDDEN_NAME = ['separator',
                  'parent',
                  '__check_loop',
                  '__detach',
                  '__attach',
                  '__children_or_empty',
                  'children',
                  '__check_children',
                  'children_setter',
                  'children_deleter',
                  '_pre_detach_children',
                  '_post_detach_children',
                  '_pre_attach_children',
                  '_post_attach_children',
                  'path',
                  'iter_path_reverse',
                  '_path',
                  'ancestors',
                  'anchestors',
                  'descendants',
                  'root',
                  'siblings',
                  'leaves',
                  'is_leaf'
                  'is_root',
                  'height',
                  'depth',
                  '_pre_detach',
                  '_post_detach',
                  '_pre_attach',
                  '_post_attach']


def name_to_safename(name: str) -> str:
    name_ = name.replace('.', '_')  # TODO this replace only one symbol. This MUST be more replacements.
    return name_


class ExtendedNode(NodeMixin):
    resolver = Resolver('name')

    def __init__(self, name, parent=None, _print_color=None, _prefix: str = 'UNDEFINED'):
        super(ExtendedNode, self).__init__()
        if name in FORBIDDEN_NAME:
            raise ValueError(f'{name} cannot be used for the space element. This name is reserved by anytree package.')

        self.name = name_to_safename(name)
        self.parent = parent
        self._print_color = _print_color
        self._prefix = _prefix

    def __len__(self):
        return len(self.children)

    def __getitem__(self, request_: Union[int, str]):

        if isinstance(request_, int):
            return self.children[request_]
        elif isinstance(request_, str):
            return self.resolver.get(self, request_)
        else:
            raise NotImplementedError()

    def __repr__(self):
        return self._print_color + f'{self._prefix}:{self.name}' + Fore.RESET

    def _post_attach(self, parent):
        if parent is not None:
            setattr(parent, self.name, self)

    def _post_detach(self, parent):
        if parent is not None:
            if hasattr(parent, self.name):
                delattr(parent, self.name)

    def explore(self):
        return RenderTree(self).__str__()


class Space(ExtendedNode):
    def __init__(self, name, parent=None):
        super(Space, self).__init__(name, parent=parent, _print_color=Fore.RED, _prefix='S')
        self.version = nndt.__version__

    def __repr__(self):
        return self._print_color + f'{self._prefix}:{self.name}' + Fore.WHITE + f' v{self.version}' + Fore.RESET

class Group(ExtendedNode):
    def __init__(self, name, parent=None):
        super(Group, self).__init__(name, parent=parent, _print_color=Fore.RED, _prefix='G')


class Object3D(ExtendedNode):
    def __init__(self, name, parent=None):
        super(Object3D, self).__init__(name, parent=parent, _print_color=Fore.BLUE, _prefix='O3D')


class FileSource(ExtendedNode):
    def __init__(self, name, filepath, parent=None):
        super(FileSource, self).__init__(name, parent=parent, _print_color=Fore.GREEN, _prefix='S')
        if not os.path.exists(filepath):
            raise FileNotFoundError()
        self.filepath = filepath

    def __repr__(self):
        return self._print_color + f'{self._prefix}:{self.name}' + Fore.WHITE + f"'{self.filepath}'" + Fore.RESET


def load_from_path(root_path):
    space = Space("space")

    def add_values(lst, fullpath):
        if len(lst) >= 2:
            current_node_ = space
            for ind, name_ in enumerate(lst):
                child_names = [child.name for child in current_node_.children]
                if name_ in child_names:
                    current_node_ = current_node_[name_]
                else:
                    if ind < (len(lst) - 2):
                        current_node_ = Group(name_, parent=current_node_)
                    elif ind == (len(lst) - 2):
                        current_node_ = Object3D(name_, parent=current_node_)
                    elif ind == (len(lst) - 1):
                        current_node_ = FileSource(name_, fullpath, parent=current_node_)

    for root, dirs, files in os.walk(root_path, topdown=False):
        for fl in files:
            line = os.path.relpath(root, root_path)
            line2 = os.path.join(line, fl)
            lst = line2.split('/')
            add_values(lst, os.path.join(root, fl))

    return space
