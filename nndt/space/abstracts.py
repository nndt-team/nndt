import os
from anytree import NodeMixin, Resolver, RenderTree
from colorama import Fore
from typing import *


class ExtendedNodeMixin(NodeMixin):
    resolver = Resolver('name')

    def __len__(self):
        return len(self.children)

    def __getitem__(self, request_: Union[int, str]):

        if isinstance(request_, int):
            return self.children[request_]
        elif isinstance(request_, str):
            return self.resolver.get(self, request_)
        else:
            raise NotImplementedError()

    def explore(self):
        return RenderTree(self)


class AbstractRegion:

    def __init__(self,
                 _ndim=3,
                 _bbox=((0., 0., 0.), (0., 0., 0.)),
                 name=""):
        self._ndim = _ndim
        self._bbox = _bbox
        self.name = name
        self._print_color = Fore.BLUE

    def __print_bbox(self):
        a = self._bbox
        return f"(({a[0][0]:.02f}, {a[0][1]:.02f}, {a[0][2]:.02f}), ({a[1][0]:.02f}, {a[1][1]:.02f}, {a[1][2]:.02f}))"

    def __repr__(self):
        return self._print_color + f'{str(self.__class__.__name__)}("{self.name}", bbox="{self.__print_bbox()}")' + Fore.RESET


class AbstractSource:
    def __init__(self):
        self.name = ""

    def __repr__(self):
        return Fore.LIGHTBLUE_EX + f'{str(self.__class__.__name__)}("{self.name}")' + Fore.RESET


class FileSource(AbstractSource):

    def __init__(self, filepath):
        super(FileSource, self).__init__()
        if not os.path.exists(filepath):
            raise FileNotFoundError()
        self.filepath = filepath

    def __repr__(self):
        return Fore.LIGHTBLUE_EX + f'{str(self.__class__.__name__)}("{self.name}", filename="{os.path.basename(self.filepath)}")' + Fore.RESET


class AbstractMethod:

    def __init__(self):
        pass
