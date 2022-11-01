from tqdm import tqdm

from nndt import *
from nndt.space2 import *

P = {
    "lr": 0.001,
    "epoch": 9001,
    "shape": (64, 64, 64),
    "dataset_path": "../../tests/acdc_for_test",
    "flat_shape": 64 * 64 * 64,
    "viz_shape": (128, 128, 128),
    "exp_name": "infer_LipMLP_2",
    "log_folder": f"./sdt2sdf_default/",
    "preload_mode": "shift_and_scale",
    "level_shift": 0.03,
    "model_params": "./train_LipMLP_2/sdf_model.pkl",
}

if __name__ == "__main__":
    name_list = ["patient009", "patient089"]

    task = ApproximateSDF(batch_size=P["flat_shape"], model_number=len(name_list))
    rng = jax.random.PRNGKey(42)
    _, F = task.init_and_functions(rng)
    with open(P["model_params"], "rb") as fl:
        params = pickle.load(fl)

    viz = BasicVizualization(P["exp_name"], P["exp_name"], print_on_each_epoch=1000)
    xyz = grid_in_cube2(
        spacing=P["shape"], lower=(-1.0, -1.0, -1.0), upper=(1.0, 1.0, 1.0)
    )
    xyz = xyz.reshape((-1, 3))
    bary = barycentric_grid(order=(-1, 1), spacing=(0, 5), filter_negative=True)
    for c in tqdm(bary):
        c_ = jnp.insert(c, slice(2, 3), jnp.array([0.0]))

        PP = jnp.tile(c_, (xyz.shape[0], 1))
        predict_sdf = F.vec_sdf(
            params, rng, xyz[:, 0], xyz[:, 1], xyz[:, 2], jnp.zeros(xyz.shape[0]), PP
        ).reshape(P["shape"])
        viz.sdt_to_obj(f"SDF2_{c[0]}", predict_sdf, level=P["level_shift"])
