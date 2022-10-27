from nndt.space2 import *
from nndt.vizualize import save_sdt_as_obj

P = {
    "lr": 0.006,
    "epoch": 9001,
    "shape": (128, 128, 128),
    "dataset_path": "../../tests/acdc_for_test",
    "flat_shape": 64 * 64 * 64,
    "exp_name": "sdt2sdf_check",
    "log_folder": f"./sdt2sdf_default/",
    "preload_mode": "shift_and_scale",
    "level_shift": 0.03,
    "I": {"exp_name": "sdt2sdf_default", "depth": 8, "weight": 64},
}


def main():
    """
    This step applies the marching cubes to data and vizualize original and trained representation of th objects.
    :return:
    """
    IN = P["I"]
    space_orig = load_from_path(P["dataset_path"])
    space_orig.preload("shift_and_scale")
    print(space_orig.print("sources"))

    for width in [IN["weight"]]:
        for depth in [IN["depth"]]:
            path_in = f"./{IN['exp_name']}/{depth:02}_{width:02}/"
            path_out = f"./{P['exp_name']}/{depth:02}_{width:02}/"

            for patient in space_orig:
                grid_xyz = patient.sampling_grid(P["shape"])
                grid_sdt = patient.surface_xyz2sdt(grid_xyz)
                os.makedirs(path_out + f"{patient.name}", exist_ok=True)
                save_sdt_as_obj(
                    path_out + f"{patient.name}/original.obj",
                    jnp.squeeze(grid_sdt),
                    level=P["level_shift"],
                )

            space_ir = load_from_path(path_in)
            space_ir.preload("shift_and_scale")
            for patient in space_ir:
                grid_xyz = patient.sampling_grid(P["shape"])
                grid_sdt = patient.surface_xyz2sdt(grid_xyz)
                save_sdt_as_obj(
                    path_out + f"{patient.name}/imp_repr.obj",
                    jnp.squeeze(grid_sdt),
                    level=P["level_shift"],
                )

    # Open and look
    for width in [IN["weight"]]:
        for depth in [IN["depth"]]:
            for name in os.listdir(f"./{IN['exp_name']}/{depth:02}_{width:02}/"):
                mesh1 = f"./{P['exp_name']}/{depth:02}_{width:02}/{name}/original.obj"
                mesh2 = f"./{P['exp_name']}/{depth:02}_{width:02}/{name}/imp_repr.obj"
                space_viz = load_from_file_lists(
                    ["orig", "pred"], mesh_list=[mesh1, mesh2]
                )
                space_viz.preload("shift_and_scale")
                space_viz.plot()


if __name__ == "__main__":
    main()
