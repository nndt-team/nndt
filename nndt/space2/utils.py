from typing import *

import jax.numpy as jnp
import numpy as onp
from skimage import measure


def update_bbox(bbox1: ((float, float, float), (float, float, float)),
                bbox2: ((float, float, float), (float, float, float))):
    (Xmin1, Ymin1, Zmin1), (Xmax1, Ymax1, Zmax1) = bbox1
    (Xmin2, Ymin2, Zmin2), (Xmax2, Ymax2, Zmax2) = bbox2
    return ((min(Xmin1, Xmin2), min(Ymin1, Ymin2), min(Zmin1, Zmin2)),
            (max(Xmax1, Xmax2), max(Ymax1, Ymax2), max(Zmax1, Zmax2)))


def save_verts_and_faces_to_obj(filepath: str, verts, faces):
    with open(filepath, 'w') as fl:
        for v in verts:
            fl.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for f in faces:
            fl.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")


def array_to_vert_and_faces(array: Union[jnp.ndarray, onp.ndarray],
                            level: float = 0.0, for_vtk_cell_array: bool = False):
    level_ = level
    if not (array.min() < level_ < array.max()):
        level_ = (array.max() + array.min()) / 2.

    verts, faces, _, _ = measure.marching_cubes(onp.array(array), level=level_)

    if for_vtk_cell_array:
        faces = onp.concatenate([onp.full((faces.shape[0], 1), 3), faces], axis=1)
        faces = faces.flatten()

    return verts, faces
