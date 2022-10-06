import fnmatch
import os
from typing import *

from anytree import NodeMixin, Resolver, RenderTree
from anytree.exporter import JsonExporter, DictExporter
from anytree.importer import JsonImporter, DictImporter
from colorama import Fore

import nndt

NODE_METHOD_DICT = {}

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

def _children_filter(children):
    ret = [v for v in children if isinstance(v, ExtendedNode)]
    return ret

def _nodecls_function(parent=None, **attrs):
    if '_nodetype' not in attrs:
        raise ValueError('_nodetype is not located in some node of space file')

    if attrs['_nodetype'] == 'FS':
        ret = DICT_NODETYPE_CLASS[attrs['_nodetype']](attrs['name'], attrs['filepath'], attrs['loader_type'],
                                                      parent=parent)
    else:
        ret = DICT_NODETYPE_CLASS[attrs['_nodetype']](attrs['name'], parent=parent)

    return ret

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

class MethodNode(NodeMixin):
    def __init__(self, name: str, docstring: Optional[str], parent=None,
                 _print_color: str = Fore.RESET,
                 _nodetype: str = 'M'):
        super(MethodNode, self).__init__()
        if name in FORBIDDEN_NAME:
            raise ValueError(f'{name} cannot be used for the space element. This name is reserved by anytree package.')

        self.name = name
        self.docstring = docstring if docstring is not None else name
        self.parent = parent
        self._print_color = _print_color
        self._nodetype = _nodetype

    def __repr__(self):
        return self._print_color + f'{self._nodetype}:{self.docstring}' + Fore.RESET



def initialize_method_node(obj: object):
    class_hierarchy = list(obj.__class__.__bases__)
    class_hierarchy = class_hierarchy + [obj.__class__]
    for class_name in reversed([str(class_.__name__) for class_ in class_hierarchy]):
        if class_name in NODE_METHOD_DICT:
            for fn_name, fn_docs in NODE_METHOD_DICT[class_name].items():
                if hasattr(obj, fn_name) and (fn_name not in [x.name for x in obj.children]):
                    method = MethodNode(fn_name,
                                        fn_docs,
                                        parent=obj)

class ExtendedNode(NodeMixin):
    resolver = Resolver('name')

    def __init__(self, name: str,
                 parent=None,
                 bbox=((0., 0.), (0., 0.), (0., 0.)),
                 _print_color: str = None,
                 _nodetype: str = 'UNDEFINED'):
        super(ExtendedNode, self).__init__()
        if name in FORBIDDEN_NAME:
            raise ValueError(f'{name} cannot be used for the space element. This name is reserved by anytree package.')

        self.name = _name_to_safename(name)
        self.parent = parent
        self.bbox = bbox
        self._print_color = _print_color
        self._nodetype = _nodetype

    def __len__(self):
        return len(self.children)

    def __getitem__(self, request_: Union[int, str]):

        if isinstance(request_, int):
            children_without_methods = [ch for ch in self.children if isinstance(ch, ExtendedNode)]
            return children_without_methods[request_]
        elif isinstance(request_, str):
            return self.resolver.get(self, request_)
        else:
            raise NotImplementedError()

    def __repr__(self):
        return self._print_color + f'{self._nodetype}:{self.name}' + Fore.RESET

    def _print_bbox(self):
        a = self.bbox
        return f"(({a[0][0]:.02f}, {a[0][1]:.02f}), ({a[1][0]:.02f}, {a[1][1]:.02f}), ({a[2][0]:.02f}, {a[2][1]:.02f}))"

    def _post_attach(self, parent):
        if parent is not None:
            setattr(parent, self.name, self)

        initialize_method_node(self)

    def _post_detach(self, parent):
        if parent is not None:
            if hasattr(parent, self.name):
                delattr(parent, self.name)

    @node_method("do_nothing()")
    def do_nothing(self):
        pass

    @node_method("explore(default|full)")
    def explore(self, mode: Optional[str] = "default"):
        if mode is None or (mode == "default"):
            ret = RenderTree(self, childiter=_children_filter).__str__()
        elif mode == "full":
            ret = RenderTree(self).__str__()
        return ret


class Space(ExtendedNode):
    def __init__(self, name,
                 bbox=((0., 0.), (0., 0.), (0., 0.)),
                 parent=None):
        super(Space, self).__init__(name, parent=parent, bbox=bbox, _print_color=Fore.RED, _nodetype='S')
        self.version = nndt.__version__

        initialize_method_node(self)

    def save_space(self, filepath: str):
        filepath_with_ext = filepath if filepath.endswith('.space') else filepath + '.space'
        with open(filepath_with_ext, 'w', encoding='utf-8') as fl:
            fl.write(self.to_json())

    def to_json(self):
        dict_exp = DictExporter(attriter=_attribute_filter, childiter=_children_filter)
        json_exp = JsonExporter(dictexporter=dict_exp, indent=2)
        return json_exp.export(self)

    def __repr__(self):
        return self._print_color + f'{self._nodetype}:{self.name}' + Fore.WHITE + f' {self.version}' + Fore.RESET


class Group(ExtendedNode):
    def __init__(self, name,
                 bbox=((0., 0.), (0., 0.), (0., 0.)),
                 parent=None):
        super(Group, self).__init__(name, parent=parent, bbox=bbox, _print_color=Fore.RED, _nodetype='G')


class Object3D(ExtendedNode):
    def __init__(self, name,
                 bbox=((0., 0.), (0., 0.), (0., 0.)),
                 parent=None):
        super(Object3D, self).__init__(name, parent=parent, bbox=bbox, _print_color=Fore.BLUE, _nodetype='O3D')

    def __repr__(self):
        return self._print_color + f'{self._nodetype}:{self.name}' + Fore.WHITE + f' {self._print_bbox()}' + Fore.RESET


class FileSource(ExtendedNode):
    def __init__(self, name, filepath, loader_type,
                 bbox=((0., 0.), (0., 0.), (0., 0.)),
                 parent=None):
        super(FileSource, self).__init__(name, parent=parent, bbox=bbox, _print_color=Fore.GREEN, _nodetype='FS')
        if not os.path.exists(filepath):
            raise FileNotFoundError()
        self.filepath = filepath
        self.loader_type = loader_type

    def __repr__(self):
        return self._print_color + f'{self._nodetype}:{self.name}' + Fore.WHITE + f" {self.loader_type} {self.filepath}" + Fore.RESET


def load_from_path(root_path,
                   template_txt='*.txt',
                   template_sdt='*sd[ft]*.npy',
                   template_mesh_obj='*.obj'):
    space = Space("space")

    def filename_to_loader_type(filename):
        if fnmatch.fnmatch(filename, template_txt):
            ret = 'txt'
        elif fnmatch.fnmatch(filename, template_sdt):
            ret = 'sdt'
        elif fnmatch.fnmatch(filename, template_mesh_obj):
            ret = 'mesh_obj'
        else:
            ret = 'undefined'
        return ret

    def add_values(lst, fullpath, filename):
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
                        current_node_ = FileSource(name_, fullpath, filename_to_loader_type(filename),
                                                   parent=current_node_)

    for root, dirs, files in os.walk(root_path, topdown=False):
        for fl in files:
            line = os.path.relpath(root, root_path)
            line2 = os.path.join(line, fl)
            lst = line2.split('/')
            add_values(lst, os.path.join(root, fl), fl)

    return space


DICT_NODETYPE_CLASS = {'UNDEFINED': None,
                       'S': Space,
                       'G': Group,
                       'O3D': Object3D,
                       'FS': FileSource,
                       "M": MethodNode,
                       }
DICT_CLASS_NODETYPE = {(v, k) for k, v in DICT_NODETYPE_CLASS.items()}


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
