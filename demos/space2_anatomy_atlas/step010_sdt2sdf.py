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
    This step train usual MLP for representation of the models.
    :return:
    """
    space = load_from_path(P["dataset_path"])
    space.preload("shift_and_scale", ns_padding=(0.1, 0.1, 0.1))
    print(space.print())

    # for width in [2, 4, 8, 16, 32, 64]:
    #     for depth in [1, 2, 4, 8, 16, 32, 64]:

    for width in [64]:
        for depth in [8]:
            for patient in space:
                path = f"./{P['exp_name']}/{depth:02}_{width:02}/{patient.name}/"
                os.makedirs(path, exist_ok=True)
                patient.train_task_sdt2sdf(
                    path + f"imp_repr.ir1",
                    width=width,
                    depth=depth,
                )


if __name__ == "__main__":
    main()
