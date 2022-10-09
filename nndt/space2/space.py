from anytree import PostOrderIter, PreOrderIter
from anytree.exporter import DictExporter, JsonExporter
from colorama import Fore

import nndt
from nndt.space2 import AbstractBBoxNode
from nndt.space2 import Object3D, Group, update_bbox
from nndt.space2.abstracts import node_method, AbstractTreeElement


class Space(AbstractBBoxNode):
    def __init__(self, name,
                 bbox=((0., 0., 0.), (0., 0., 0.)),
                 parent=None):
        super(Space, self).__init__(name, parent=parent, bbox=bbox, _print_color=Fore.RED, _nodetype='S')
        self.version = nndt.__version__

        self._is_init = False
        self._is_preload = False

    def __repr__(self):
        return self._print_color + f'{self._nodetype}:{self.name}' + Fore.WHITE + f' {self.version}' + Fore.RESET

    @node_method("save_space(filepath)")
    def save_space(self, filepath: str):
        from nndt.space2 import save_space
        return save_space(self, filepath)

    @node_method("to_json()")
    def to_json(self):
        from nndt.space2 import to_json
        return to_json(self)

    @node_method("init()")
    def init(self):
        for node in PostOrderIter(self):
            if isinstance(node, AbstractTreeElement):
                node._add_method_node()

    @node_method("preload(ident|shift_and_scale|to_cube, scale, keep_in_memory=True)")
    def preload(self, mode="ident", scale=50, keep_in_memory=True):
        if not self._is_preload:
            from space2.preloader import DefaultPreloader
            self.preloader = DefaultPreloader(mode=mode, scale=scale, keep_in_memory=keep_in_memory)
            self.preloader.preload(self)
