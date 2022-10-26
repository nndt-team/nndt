from nndt.space2 import *

P = {
    "lr": 0.006,
    "epoch": 9001,
    "shape": (64, 64, 64),
    "dataset_path": "../../tests/acdc_for_test",
    "flat_shape": 64 * 64 * 64,
    "exp_name": "sdt2sdf_default",
    "log_folder": f"./sdt2sdf_default/",
    "preload_mode": "shift_and_scale",
    "level_shift": 0.03,
}

if __name__ == "__main__":
    space = load_from_path(P["dataset_path"])
    space.preload("shift_and_scale")
    print(space.print())

    for width in [2, 4, 8, 16, 32, 64]:
        for depth in [1, 2, 4, 8, 16, 32, 64]:
            for patient in space:
                os.makedirs(f"./{P['exp_name']}/{patient.name}/", exist_ok=True)
                patient.train_task_sdt2sdf(
                    f"./{P['exp_name']}/{patient.name}/{depth:02}_{width:02}.ir",
                    width=width,
                    depth=depth,
                )
