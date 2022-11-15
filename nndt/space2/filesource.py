import os

from colorama import Fore
from nndt.vizualize import ANSIConverter

from nndt.space2.abstracts import AbstractBBoxNode, IterAccessMixin


class FileSource(AbstractBBoxNode, IterAccessMixin):
    """
    This class keeps the location of the file for processing.

    Args:
        name (str): name of the node
        filepath (str): a file path. If it does not exist raise FileNotFoundError.
        loader_type (str): loader type, this string notes type of information for uploading
        bbox (tuple, optional): boundary box in form ((X_min, Y_min, Z_min), (X_max, Y_max, Z_max)).
                                Defaults to ((0., 0., 0.), (0., 0., 0.)).
        parent (_type_, optional): parent node. Defaults to None.

    Raises:
        FileNotFoundError: file or directory is requested but doesn't exist.
    """

    def __init__(
        self,
        name,
        filepath: str,
        loader_type: str,
        bbox=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        parent=None,
    ):

        super(FileSource, self).__init__(
            name, parent=parent, bbox=bbox, _print_color=Fore.GREEN, _nodetype="FS"
        )
        if not os.path.exists(filepath):
            raise FileNotFoundError()
        self.filepath = filepath
        self.loader_type = loader_type
        self._loader = None

    def __repr__(self):
        star_bool = self._loader.is_load if self._loader is not None else False
        star = "^" if star_bool else ""
        return (
            self._print_color
            + f"{self._nodetype}:{self.name}"
            + Fore.WHITE
            + f" {self.loader_type}{star} {self.filepath}"
            + Fore.RESET
        )

    def _repr_html_(self):
        star_bool = self._loader.is_load if self._loader is not None else False
        star = "^" if star_bool else ""
        return (
            '<p>'
            + f'<span style=\"color:{ANSIConverter(self._print_color, type="Fore").to_rgb()}\">'
            + f'{self._nodetype}:{self.name}'
            + '</span>'
            + f'<span style=\"color:{ANSIConverter(Fore.WHITE, type="Fore").to_rgb()}\">'
            + f' {self.loader_type}{star} {self.filepath}'
            + '</span>'
            + '</p>'
        )
