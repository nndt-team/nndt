import optax

from nndt import *
from nndt.space2 import *

P = {
    "lr": 0.01,
    "epoch": 8601,
    "shape": (64, 64, 64),
    "dataset_path": "../../tests/acdc_for_test",
    "flat_shape": 64 * 64 * 64,
    "exp_name": "train_LipMLP_2",
    "log_folder": f"./sdt2sdf_default/",
    "preload_mode": "shift_and_scale",
    "level_shift": 0.03,
    "I": {"exp_name": "sdt2sdf_default", "depth": 8, "weight": 64},
}


class DataGen:
    def __init__(self, space, name_list, spacing):
        self.space = space
        self.name_list = name_list
        self.spacing = spacing
        self.space = space

    def load_batch(self):
        num_of_obj = len(self.name_list)
        batch = None

        xyz = self.space.sampling_grid(spacing=self.spacing)
        xyz_flat = xyz.reshape((-1, 3))
        for code, patient in enumerate(self.space):
            if patient.name in self.name_list:
                sdf_flat = jnp.squeeze(patient.surface_xyz2sdt(xyz_flat))

                p_array = jnp.array(jnp.zeros((sdf_flat.shape[0], num_of_obj)))
                p_array = p_array.at[:, code].set(1.0)

                DATA = ApproximateSDF.DATA(
                    X=xyz_flat[:, 0],
                    Y=xyz_flat[:, 1],
                    Z=xyz_flat[:, 2],
                    T=jnp.zeros(sdf_flat.shape[0]),
                    P=p_array,
                    SDF=sdf_flat,
                )
                if batch is None:
                    batch = DATA
                else:
                    batch = batch + DATA

        return batch


if __name__ == "__main__":
    name_list = ["patient009", "patient089"]

    task = ApproximateSDF(batch_size=P["flat_shape"], model_number=len(name_list))
    rng = jax.random.PRNGKey(42)
    params, F = task.init_and_functions(rng)
    D_INIT = task.init_data()

    opt = optax.adam(P["lr"])
    opt_state = opt.init(params)

    # Batch
    space_orig = load_from_path(P["dataset_path"])
    space_orig.preload("shift_and_scale", ns_padding=(0.1, 0.1, 0.1))
    print(space_orig.print("sources"))

    # Batch
    gen = DataGen(space_orig, name_list, spacing=P["shape"])
    D1 = gen.load_batch()

    @jax.jit
    def train_step(params, rng, opt_state, D1):

        loss, grads = jax.value_and_grad(F.vec_main_loss)(params, rng, *tuple(D1))
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return loss, params, rng, opt_state

    max_loss = 99999
    viz = BasicVizualization(P["exp_name"], P["exp_name"], print_on_each_epoch=200)
    for epoch in viz.iter(P["epoch"]):

        loss, params, rng, opt_state = train_step(params, rng, opt_state, D1)
        viz.record({"loss": float(loss)})

        if viz.is_print_on_epoch(epoch):
            viz.draw_loss("TRAIN_LOSS", viz._records["loss"])
            if loss < max_loss:
                viz.save_state("sdf_model", params)
                max_loss = loss

        rng, subkey = jax.random.split(rng)

    viz.draw_loss("TRAIN_LOSS", viz._records["loss"])
