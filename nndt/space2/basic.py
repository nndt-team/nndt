import json
import os
from typing import *

from anytree import NodeMixin, Resolver, RenderTree
from anytree.exporter import JsonExporter, DictExporter
from anytree.importer import JsonImporter, DictImporter
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


def _name_to_safename(name: str) -> str:
    name_ = name.replace('.', '_')  # TODO this replace only one symbol. This MUST be more replacements.
    return name_


def _attribute_filter(attrs):
    ret = [(k, v) for k, v in attrs if isinstance(v, (int, float, str, tuple))]
    return sorted(ret)


def _nodecls_function(parent=None, **attrs):
    if '_nodetype' not in attrs:
        raise ValueError('_nodetype is not located in some node of space file')

    if attrs['_nodetype'] == 'FS':
        ret = DICT_NODETYPE_CLASS[attrs['_nodetype']](attrs['name'], attrs['filepath'], parent=parent)
    else:
        ret = DICT_NODETYPE_CLASS[attrs['_nodetype']](attrs['name'], parent=parent)

    return ret

def load_space(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError()

    dict_imp = DictImporter(nodecls=_nodecls_function)
    json_imp = JsonImporter(dictimporter=dict_imp)
    with open(filepath, 'r') as fl:
        space = json_imp.read(fl)

    return space
def from_json(json: str):
    dict_imp = DictImporter(nodecls=_nodecls_function)
    json_imp = JsonImporter(dictimporter=dict_imp)
    space = json_imp.import_(json)
    return space

class ExtendedNode(NodeMixin):
    resolver = Resolver('name')

    def __init__(self, name, parent=None, _print_color=None, _nodetype: str = 'UNDEFINED'):
        super(ExtendedNode, self).__init__()
        if name in FORBIDDEN_NAME:
            raise ValueError(f'{name} cannot be used for the space element. This name is reserved by anytree package.')

        self.name = _name_to_safename(name)
        self.parent = parent
        self._print_color = _print_color
        self._nodetype = _nodetype

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
        return self._print_color + f'{self._nodetype}:{self.name}' + Fore.RESET

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
        super(Space, self).__init__(name, parent=parent, _print_color=Fore.RED, _nodetype='S')
        self.version = nndt.__version__

    def save_space(self, filepath: str):
        filepath_with_ext = filepath if filepath.endswith('.space') else filepath + '.space'
        with open(filepath_with_ext, 'w', encoding='utf-8') as fl:
            fl.write(self.to_json())

    def to_json(self):
        dict_exp = DictExporter(attriter=_attribute_filter)
        json_exp = JsonExporter(dictexporter=dict_exp, indent=2)
        return json_exp.export(self)

    def __repr__(self):
        return self._print_color + f'{self._nodetype}:{self.name}' + Fore.WHITE + f' v{self.version}' + Fore.RESET


class Group(ExtendedNode):
    def __init__(self, name, parent=None):
        super(Group, self).__init__(name, parent=parent, _print_color=Fore.RED, _nodetype='G')


class Object3D(ExtendedNode):
    def __init__(self, name, parent=None):
        super(Object3D, self).__init__(name, parent=parent, _print_color=Fore.BLUE, _nodetype='O3D')


class FileSource(ExtendedNode):
    def __init__(self, name, filepath=None, parent=None):
        super(FileSource, self).__init__(name, parent=parent, _print_color=Fore.GREEN, _nodetype='FS')
        if not os.path.exists(filepath):
            raise FileNotFoundError()
        self.filepath = filepath

    def __repr__(self):
        return self._print_color + f'{self._nodetype}:{self.name}' + Fore.WHITE + f"'{self.filepath}'" + Fore.RESET


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


DICT_NODETYPE_CLASS = {'UNDEFINED': None,
                       'S': Space,
                       'G': Group,
                       'O3D': Object3D,
                       'FS': FileSource
                       }
DICT_CLASS_NODETYPE = {(v, k) for k, v in DICT_NODETYPE_CLASS.items()}
