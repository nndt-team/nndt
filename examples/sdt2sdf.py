import os

import jax.numpy as jnp

from nndt import *
from nndt.space2 import *

CONF = {
    "lr": 0.005,
    "epoch": 10001,
    "width": 64,
    "depth": 8,
    "shape": (64, 64, 64),
    "shape_viz": (128, 128, 128),
    "dataset_path": "../tests/acdc_for_test",
    "exp_name": "sdt2sdf",
    "level_shift": 0.03,
}


def main():
    """
    This example train usual MLP for representation of the models.
    :return:
    """
    space = load_from_path(CONF["dataset_path"])
    space.preload("shift_and_scale", ns_padding=(0.1, 0.1, 0.1))
    print(space.print("source"))

    for patient in space:
        path = f"./{CONF['exp_name']}/{patient.name}/"
        os.makedirs(path, exist_ok=True)
        patient.train_task_sdt2sdf(
            path + f"imp_repr.ir1",
            width=CONF["width"],
            depth=CONF["depth"],
            epochs=CONF["epoch"],
        )

        grid_xyz = patient.sampling_grid(CONF["shape_viz"])
        grid_sdt = patient.surface_xyz2sdt(grid_xyz)
        save_sdt_as_obj(
            jnp.squeeze(grid_sdt), path + f"original.obj", level=CONF["level_shift"]
        )

    space_ir = load_from_path(f"./{CONF['exp_name']}", template_mesh_obj=None)
    space_ir.preload("identity")
    for patient in space_ir:
        path = f"./{CONF['exp_name']}/{patient.name}/"
        grid_xyz = patient.sampling_grid(CONF["shape_viz"])
        grid_sdt = patient.surface_xyz2sdt(grid_xyz)
        save_sdt_as_obj(
            jnp.squeeze(grid_sdt), path + f"imp_repr.obj", level=CONF["level_shift"]
        )


if __name__ == "__main__":
    main()
