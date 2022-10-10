import warnings
from typing import *

import numpy as onp
import pyvista as pv
from anytree import PostOrderIter, PreOrderIter
from pyvista import Plotter

from nndt.space2 import AbstractTreeElement, AbstractBBoxNode, AbstractTransformation, FileSource, \
    MeshObjLoader, SDTLoader,  Object3D


def _plot_pv_mesh(pl: Plotter, verts, faces, transform):
    verts = transform(verts)
    poly_data = pv.PolyData(var_inp=onp.array(verts), faces=faces)
    pl.add_mesh(poly_data)


def _plot_mesh(pl: Plotter, loader: MeshObjLoader, transform):
    obj_mesh = loader.mesh
    pv_mesh = pv.PolyData(obj_mesh)
    verts = pv_mesh.points
    faces = pv_mesh.faces
    _plot_pv_mesh(pl, verts, faces, transform)


def _plot_sdt(pl: Plotter, loader: SDTLoader, transform: Callable):
    sdt = loader.sdt
    from space2 import array_to_vert_and_faces
    verts, faces = array_to_vert_and_faces(sdt, level=0.0, for_vtk_cell_array=True)
    _plot_pv_mesh(pl, verts, faces, transform)


def _plot_filesource(pl, node: FileSource, transform: Callable):
    if isinstance(node._loader, MeshObjLoader):
        _plot_mesh(pl, node._loader, transform)
    elif isinstance(node._loader, SDTLoader):
        _plot_sdt(pl, node._loader, transform)
    else:
        warnings.warn(f"Method .plot() is not implemented for {node._loader.__class__.__name__}")


def _plot(node: AbstractTreeElement,
          mode: Optional[str] = "default",
          filepath: Optional[str] = None, cpos=None,
          level: float = 0.0):
    if filepath is None:
        pl = pv.Plotter()
    else:
        pl = pv.Plotter(off_screen=True)

    default_transform = lambda xyz: xyz

    # When .plot() is called from FileSource, it draws only one object without any transformation
    if isinstance(node, FileSource):
        _plot_filesource(pl, node, default_transform)
    # When .plot() is called from region or object, it iterates over all Object3D in tree
    elif isinstance(node, AbstractBBoxNode):
        for node_obj in PreOrderIter(node):
            if isinstance(node_obj, Object3D):
                # Try to obtain ps2ns transformation
                transform_list = [child for child in node_obj.children
                                  if isinstance(child, AbstractTransformation)]
                if len(transform_list) > 0:
                    transform = transform_list[0].transform_xyz_ps2ns
                else:
                    transform = default_transform

                # Run over filesources
                for node_src in PostOrderIter(node_obj):
                    if isinstance(node_src, FileSource):
                        _plot_filesource(pl, node_src, transform)

    if filepath is None:
        pl.show(cpos=cpos)
    else:
        pl.show(screenshot=filepath, cpos=cpos)
