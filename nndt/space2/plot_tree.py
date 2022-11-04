import copy
import warnings
from typing import *

import matplotlib as mpl
import numpy as onp
import pyvista as pv
from anytree import PostOrderIter, PreOrderIter
from pyvista import Plotter

from nndt.global_config import PYVISTA_PRE_PARAMS
from nndt.space2.abstracts import AbstractBBoxNode, AbstractTreeElement
from nndt.space2.filesource import FileSource
from nndt.space2.loader import MeshObjLoader, SDTLoader
from nndt.space2.object3D import Object3D
from nndt.space2.transformation import AbstractTransformation


def _plot_pv_mesh(pl: Plotter, verts, faces, transform, color):
    verts = transform(verts)
    poly_data = pv.PolyData(var_inp=onp.array(verts), faces=faces)
    pl.add_mesh(poly_data, color=color)


def _plot_mesh(pl: Plotter, loader: MeshObjLoader, transform, color):
    obj_mesh = loader.mesh
    pv_mesh = pv.PolyData(obj_mesh)
    verts = pv_mesh.points
    faces = pv_mesh.faces
    _plot_pv_mesh(pl, verts, faces, transform, color)


def _plot_sdt(pl: Plotter, loader: SDTLoader, transform: Callable, color):
    sdt = loader.sdt
    from nndt.space2.utils import array_to_vert_and_faces

    verts, faces = array_to_vert_and_faces(sdt, level=0.0, for_vtk_cell_array=True)
    _plot_pv_mesh(pl, verts, faces, transform, color)


def _plot_filesource(pl, node: FileSource, transform: Callable, color):
    if isinstance(node._loader, MeshObjLoader):
        _plot_mesh(pl, node._loader, transform, color)
    elif isinstance(node._loader, SDTLoader):
        _plot_sdt(pl, node._loader, transform, color)
    else:
        warnings.warn(f"node._loader is None or unknown. Something goes wrong.")


def _plot(
    node: AbstractTreeElement,
    mode: Optional[str] = "default",
    filepath: Optional[str] = None,
    cpos=None,
    cmap: str = "Set3",
    **kwargs,
):
    if not node.root._is_preload:
        warnings.warn(
            "This space model is not preloaded. Output image is empty. Call .preload() from the root node to fix!"
        )

    plotter_params_ = copy.deepcopy(PYVISTA_PRE_PARAMS)
    plotter_params_.update(kwargs)

    if isinstance(cmap, str):
        cmap = mpl.cm.get_cmap(cmap)
    cmap_index = 0

    if filepath is None:
        pl = pv.Plotter(**plotter_params_)
    else:
        pl = pv.Plotter(off_screen=True, **plotter_params_)

    default_transform = lambda xyz: xyz

    # When .plot() is called from FileSource, it draws only one object without any transformation
    if isinstance(node, FileSource):
        _plot_filesource(pl, node, default_transform, cmap(cmap_index % cmap.N))
    # When .plot() is called from a region or object, it iterates over all Object3D in the tree
    elif isinstance(node, AbstractBBoxNode):
        for node_obj in PreOrderIter(node):
            if isinstance(node_obj, Object3D):
                # Try to obtain transformation from physical space to normalized space
                transform_list = [
                    child
                    for child in node_obj.children
                    if isinstance(child, AbstractTransformation)
                ]
                if len(transform_list) > 0:
                    transform = transform_list[0].transform_xyz_ps2ns
                else:
                    transform = default_transform

                # Run over the filesources
                for node_src in PostOrderIter(node_obj):
                    if isinstance(node_src, FileSource):
                        _plot_filesource(
                            pl, node_src, transform, cmap(cmap_index % cmap.N)
                        )
                        cmap_index += 1

    if filepath is None:
        pl.show(cpos=cpos)
    else:
        pl.show(screenshot=filepath, cpos=cpos)
