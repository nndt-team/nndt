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


def main():
    """
    Minimal example that is extracted from the other files.
    :return:
    """
    space = load_from_path(P["dataset_path"])
    space.preload("shift_and_scale")
    print(space.print())

    for width in [2, 4, 8, 16, 32, 64]:
        for depth in [1, 2, 4, 8, 16, 32, 64]:
            patient = space.patient009
            path = f"./{P['exp_name']}/{depth:02}_{width:02}/{patient.name}"
            os.makedirs(path, exist_ok=True)
            patient.train_task_sdt2sdf(
                path + "/imp_repr.ir1",
                width=width,
                depth=depth,
            )


if __name__ == "__main__":
    main()
