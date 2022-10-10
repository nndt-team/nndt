import warnings
from typing import Optional

from anytree import RenderTree

from space2 import AbstractTreeElement, AbstractBBoxNode, MethodSetNode, AbstractTransformation, FileSource, \
    MeshObjLoader

import pyvista as pv

def _plot(node: AbstractTreeElement,
          mode: Optional[str] = "default",
          filepath: Optional[str] = None, cpos=None,
          level: float = 0.0):
    if filepath is None:
        pl = pv.Plotter()
    else:
        pl = pv.Plotter(off_screen=True)

    if isinstance(node, FileSource):
        if isinstance(node._loader, MeshObjLoader):
            obj_mesh = node._loader.mesh
            pl.add_mesh(obj_mesh)
        else:
            warnings.warn("3D plot is unavailable for this node.")
    else:
        warnings.warn("3D plot is unavailable for this node.")

    if filepath is None:
        pl.show(cpos=cpos)
    else:
        pl.show(screenshot=filepath, cpos=cpos)