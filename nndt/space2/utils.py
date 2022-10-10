from typing import *
from skimage import measure

import numpy as onp
import jax.numpy as jnp

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
                            level: float = 0.0):
    level_ = level
    if not (array.min() < level_ < array.max()):
        level_ = (array.max() + array.min()) / 2.

    verts, faces, _, _ = measure.marching_cubes(array, level=level_)
    return verts, faces