import os

import jax
import jax.numpy as jnp
import optax
from jax import lax

from nndt import *
from nndt.space2 import *

P = {
    "name_list": ["patient009", "patient089"],  # name of model for shape interpolation
    "lr": 0.01,  # learning rate
    "epoch": 100001,  # number of training iterations
    "shape": (32, 32, 32),  # shape of sampling grid
    "flat_shape": 32 * 32 * 32,  # shape of sampling grid after the flatten
    "shape_viz": (128, 128, 128),  # shape of sampling grid for visualization
    "dataset_path": "../tests/acdc_for_test",  #
    "exp_name": "shape_regression",
    "log_folder": f"./shape_regression/",
    "ns_padding": (0.1, 0.1, 0.1),  # padding in normalized space from object boundaries
    "level_shift": 0.06,  # shift of the SDF for visualization purposes
    "pixel_surroundings_number": 8,  # criterion for the thin surfaces
    "lip_alpha": 0.0000001,  # Lipschitz regularization constant
    "sigma": 0.01,  # standart deviation for random shift of the grid sampling
}

# Kernel that counts number of the surrounding pixels
KERNEL = jnp.array(
    [
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    ],
    dtype=jnp.float32,
)[:, :, :, jnp.newaxis, jnp.newaxis]

# Data generator
class DataGen:
    def __init__(self, space, name_list, spacing):
        self.space = space
        self.name_list = name_list
        self.spacing = spacing
        self.space = space

    def load_batch(self, key):
        num_of_obj = len(self.name_list)

        X_list = []
        Y_list = []
        Z_list = []
        T_list = []
        P_list = []
        SDF_list = []
        WEIGHT_list = []

        # Sampling of data from shapes that defined as a signed distance tensors (SDT)
        xyz = self.space.sampling_grid_with_noise(
            key, spacing=self.spacing, sigma=P["sigma"]
        )
        xyz_flat = xyz.reshape((-1, 3))

        for code, patient in enumerate(self.space):
            if patient.name in self.name_list:

                # Weight map improves training on the thin elements
                sdf_flat = jnp.squeeze(patient.surface_xyz2sdt(xyz_flat))
                WEIGHT = (sdf_flat.reshape(P["shape"]) < 0.0).astype(jnp.float32)[
                    jnp.newaxis, :, :, :, jnp.newaxis
                ]
                dn = lax.conv_dimension_numbers(
                    WEIGHT.shape, KERNEL.shape, ("NHWDC", "HWDIO", "NHWDC")
                )
                conv_out = lax.conv_general_dilated(
                    WEIGHT, KERNEL, (1, 1, 1), "SAME", (1, 1, 1), (1, 1, 1), dn
                )
                nearest_ = P["pixel_surroundings_number"]
                WEIGHT = WEIGHT + 8 * WEIGHT * (jnp.abs(conv_out) <= nearest_)
                WEIGHT += 1
                WEIGHT = jnp.squeeze(WEIGHT).reshape(P["flat_shape"])

                p_array = jnp.array(jnp.zeros((sdf_flat.shape[0], num_of_obj)))
                p_array = p_array.at[:, code].set(1.0)

                X_list.append(xyz_flat[:, 0])
                Y_list.append(xyz_flat[:, 1])
                Z_list.append(xyz_flat[:, 2])
                T_list.append(jnp.zeros(sdf_flat.shape[0]))
                P_list.append(p_array)
                SDF_list.append(sdf_flat)
                WEIGHT_list.append(WEIGHT)

        DATA = ApproximateSDFLipMLP2.DATA(
            X=jnp.concatenate(X_list, axis=0),
            Y=jnp.concatenate(Y_list, axis=0),
            Z=jnp.concatenate(Z_list, axis=0),
            T=jnp.concatenate(T_list, axis=0),
            P=jnp.concatenate(P_list, axis=0),
            SDF=jnp.concatenate(SDF_list, axis=0),
            WEIGHT=jnp.concatenate(WEIGHT_list, axis=0),
        )

        return DATA


def main():
    os.makedirs(P["exp_name"], exist_ok=True)
    task = ApproximateSDFLipMLP2(
        batch_size=P["flat_shape"],
        model_number=len(P["name_list"]),
        lip_alpha=P["lip_alpha"],
    )
    rng = jax.random.PRNGKey(42)
    params, F = task.init_and_functions(rng)
    D_INIT = task.init_data()

    linear_decay_scheduler = optax.piecewise_constant_schedule(
        init_value=P["lr"],
        boundaries_and_scales={5000: 0.6, 10000: 0.6, 20000: 0.8, 40000: 0.8},
    )
    opt = optax.adam(linear_decay_scheduler)
    opt_state = opt.init(params)

    # Load data
    space_orig = load_from_path(P["dataset_path"])
    space_orig.preload("shift_and_scale", ns_padding=P["ns_padding"])
    print(space_orig.print())

    # Load the first batch and check weight maps
    gen = DataGen(space_orig, P["name_list"], spacing=P["shape"])
    rng, subkey = jax.random.split(rng)
    D1 = gen.load_batch(subkey)
    for i, name in enumerate(P["name_list"]):
        start = i * P["flat_shape"]
        finish = (i + 1) * P["flat_shape"]
        save_3D_slices(
            D1.WEIGHT[start:finish, ...].reshape(P["shape"]),
            P["log_folder"] + f"weight_map_{name}.png",
            slice_num=7,
            levels=(),
            level_colors=(),
        )

    @jax.jit
    def train_step(params, rng, opt_state, D1):

        loss, grads = jax.value_and_grad(F.vec_main_loss)(params, rng, *tuple(D1))
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return loss, params, rng, opt_state

    # This object works like a simple experiment tracker
    max_loss = 99999
    viz = BasicVizualization(P["exp_name"], P["exp_name"], print_on_each_epoch=5000)

    # Preparation of iterator in the simplex
    bary = barycentric_grid(order=(-1, 1), spacing=(0, 5), filter_negative=True)

    # Load sampling for the visualization purposes
    xyz = grid_in_cube2(
        spacing=P["shape_viz"], lower=(-1.0, -1.0, -1.0), upper=(1.0, 1.0, 1.0)
    )
    for i, name in enumerate(P["name_list"]):
        std = space_orig[name].surface_xyz2sdt(xyz)
        save_3D_slices(
            std,
            P["log_folder"] + f"expected_{name}.png",
            slice_num=7,
            levels=(),
            level_colors=(),
        )
    xyz = xyz.reshape((-1, 3))

    # The main training cycle
    for epoch in viz.iter(P["epoch"]):

        loss, params, rng, opt_state = train_step(params, rng, opt_state, D1)
        viz.record({"loss": float(loss)})

        rng, subkey = jax.random.split(rng)
        D1 = gen.load_batch(subkey)

        if (epoch % 5000) == 0 or (epoch > 0.95 * P["epoch"]):
            if loss < max_loss:
                viz.save_state("sdf_model", params)
                max_loss = loss
                viz.draw_loss("TRAIN_LOSS", viz._records["loss"])

                for c in bary:
                    PP = jnp.tile(jnp.array(c), (xyz.shape[0], 1))
                    predict_sdf = F.vec_sdf(
                        params,
                        rng,
                        xyz[:, 0],
                        xyz[:, 1],
                        xyz[:, 2],
                        jnp.zeros(xyz.shape[0]),
                        PP,
                    ).reshape(P["shape_viz"])

                    viz.sdt_to_obj(
                        f"pred_6_{c[0]}.obj", predict_sdf, level=P["level_shift"]
                    )
                    save_3D_slices(predict_sdf, P["log_folder"] + f"pred_{c[0]}.png")

    viz.draw_loss("TRAIN_LOSS", viz._records["loss"])


if __name__ == "__main__":
    main()
