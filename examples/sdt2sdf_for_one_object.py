import os

import jax.numpy as jnp

from nndt import *
from nndt.space2 import *

P = {
    "lr": 0.005,  # learning rate
    "epoch": 10001,  # number of training iterations
    "width": 64,  # number of the perceptron in the deep layers
    "depth": 8,  # number of deep layers
    "shape": (64, 64, 64),  # shape of sampling grid
    "shape_viz": (128, 128, 128),  # shape of sampling grid for visualization
    "ns_padding": (0.1, 0.1, 0.1),  # padding in normalized space from object boundaries
    "dataset_path": "../tests/acdc_for_test",  # path to dataset
    "exp_name": "sdt2sdf",  # name of experiment folder
    "level_shift": 0.03,  # shift of the SDF for visualization purposes
}


def main():
    """
    This example train usual MLP for representation of the models.
    """
    space = load_from_path(P["dataset_path"])
    space.preload("shift_and_scale", ns_padding=P["ns_padding"])
    print(space.print("source"))

    # Train MLP and save it as a file
    for patient in space:
        path = f"./{P['exp_name']}/{patient.name}/"
        os.makedirs(path, exist_ok=True)
        patient.train_task_sdt2sdf(
            path + f"imp_repr.ir1",
            width=P["width"],
            depth=P["depth"],
            epochs=P["epoch"],
        )

        grid_xyz = patient.sampling_grid(P["shape_viz"])
        grid_sdt = patient.surface_xyz2sdt(grid_xyz)
        save_sdt_as_obj(
            jnp.squeeze(grid_sdt), path + f"original.obj", level=P["level_shift"]
        )

    # Load MLP and check results
    space_ir = load_from_path(f"./{P['exp_name']}", template_mesh_obj=None)
    space_ir.preload("identity")
    for patient in space_ir:
        path = f"./{P['exp_name']}/{patient.name}/"
        grid_xyz = patient.sampling_grid(P["shape_viz"])
        grid_sdt = patient.surface_xyz2sdt(grid_xyz)
        save_sdt_as_obj(
            jnp.squeeze(grid_sdt), path + f"imp_repr.obj", level=P["level_shift"]
        )


if __name__ == "__main__":
    main()
