import warnings
from typing import Optional

from anytree import RenderTree

from space2 import AbstractTreeElement, AbstractBBoxNode, MethodSetNode, AbstractTransformation, FileSource, \
    MeshObjLoader, SDTLoader, array_to_vert_and_faces

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
        elif isinstance(node._loader, SDTLoader):
            sdt = node._loader.sdt
            verts, faces = array_to_vert_and_faces(sdt, level=0.0, for_vtk_cell_array=True)
            poly_data = pv.PolyData(var_inp=verts, faces=faces)
            pl.add_mesh(poly_data)
        else:
            warnings.warn("3D plot is unavailable for this node.")
    else:
        warnings.warn("3D plot is unavailable for this node.")

    if filepath is None:
        pl.show(cpos=cpos)
    else:
        pl.show(screenshot=filepath, cpos=cpos)