import pickle

import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from nndt.math_core import barycentric_grid, grid_in_cube2
from nndt.space.loaders import load_data, preload_all_possible
from nndt.trainable_task import ApproximateSDFLipMLP
from nndt.vizualize import BasicVizualization

LEARNING_RATE = 0.001
EPOCHS = 20001
SHAPE = (128, 128, 128)
FLAT_SHAPE = SHAPE[0] * SHAPE[1] * SHAPE[2]
LOAD_DATA = "./shape_interpolation_LipMLP_2/sdf_model.pkl"
EXP_NAME = "shape_interpolation_viz_LipMLP_2"
LOG_FOLDER = f"./{EXP_NAME}/"
LEVEL_SHIFT = 0.09


if __name__ == "__main__":
    # NN initialization
    task = ApproximateSDFLipMLP(batch_size=FLAT_SHAPE, model_number=5)
    rng = jax.random.PRNGKey(42)
    _, F = task.init_and_functions(rng)
    with open(LOAD_DATA, "rb") as fl:
        params = pickle.load(fl)

    viz = BasicVizualization(LOG_FOLDER, EXP_NAME, print_on_each_epoch=100)

    xyz = grid_in_cube2(spacing=SHAPE, lower=(-1.0, -1.0, -1.0), upper=(1.0, 1.0, 1.0))
    xyz = xyz.reshape((-1, 3))

    bary = barycentric_grid(order=(-1, 1), spacing=(0, 5), filter_negative=True)
    for c in tqdm(bary):
        c_ = jnp.insert(c, slice(2, 3), jnp.array([0.0]))  # TODO

        P = jnp.tile(c_, (xyz.shape[0], 1))
        predict_sdf = F.vec_sdf(
            params, rng, xyz[:, 0], xyz[:, 1], xyz[:, 2], jnp.zeros(xyz.shape[0]), P
        ).reshape(SHAPE)
        if (c[0] == 0) or (c[0] == 1):
            viz.sdt_to_obj(f"SDF2_{c[0]}", predict_sdf, level=0.06)
        else:
            viz.sdt_to_obj(f"SDF2_{c[0]}", predict_sdf, level=0.09)
