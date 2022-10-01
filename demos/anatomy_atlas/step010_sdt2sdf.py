import os

import jax
import jax.numpy as jnp
import optax

from nndt.math_core import grid_in_cube2
from nndt.space.loaders import load_data, preload_all_possible
from nndt.trainable_task import ApproximateSDF
from nndt.vizualize import BasicVizualization

LEARNING_RATE = 0.006
EPOCHS = 9001
SHAPE = (64, 64, 64)
FLAT_SHAPE = SHAPE[0] * SHAPE[1] * SHAPE[2]
EXP_NAME = 'sdt2sdf_default'
LOG_FOLDER = f'./{EXP_NAME}/'
LEVEL_SHIFT = 0.03


if __name__ == '__main__':

    os.makedirs(LOG_FOLDER, exist_ok=True)

    folder = '../../tests/acdc_for_test'
    name_list = os.listdir(folder)
    name_list.sort()
    mesh_list = [f"{folder}/{p}/colored.obj" for p in name_list]
    sdt_list = [f"{folder}/{p}/sdf.npy" for p in name_list]
    space = load_data(name_list, mesh_list, sdt_list)
    preload_all_possible(space)
    print(space.explore())

    for width in [2, 4, 8, 16, 32, 64]:
        for depth in [1, 2, 4, 8, 16, 32, 64]:
            for patient in name_list[:1]:
                train = space[f'default/{patient}/sdt/repr/train_sdt2sdf']
                train(f'./{EXP_NAME}/{depth}_{width}_{patient}', width=width, depth=depth)