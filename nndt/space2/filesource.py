import os

from colorama import Fore

from nndt.space2.abstracts import AbstractBBoxNode, IterAccessMixin


class FileSource(AbstractBBoxNode, IterAccessMixin):
    def __init__(self, name, filepath: str, loader_type: str,
                 bbox=((0., 0., 0.), (0., 0., 0.)),
                 parent=None):
        super(FileSource, self).__init__(name, parent=parent, bbox=bbox, _print_color=Fore.GREEN, _nodetype='FS')
        if not os.path.exists(filepath):
            raise FileNotFoundError()
        self.filepath = filepath
        self.loader_type = loader_type
        self._loader = None

    def __repr__(self):
        star_bool = self._loader.is_load if self._loader is not None else False
        star = "^" if star_bool else ""
        return self._print_color + f'{self._nodetype}:{self.name}' + \
               Fore.WHITE + f" {self.loader_type}{star} {self.filepath}" + Fore.RESET
