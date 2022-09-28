import jax
import jax.numpy as jnp
import optax

from nndt.space.loaders import load_data, preload_all_possible
from nndt.trainable_task import ApproximateSDFLipMLP
from nndt.vizualize import BasicVizualization

LEARNING_RATE = 0.005
EPOCHS = 20001
SHAPE = (64, 64, 64)
FLAT_SHAPE = SHAPE[0] * SHAPE[1] * SHAPE[2]
EXP_NAME = 'sdf_multiple_files_LipMLP'
LOG_FOLDER = f'./{EXP_NAME}/'
LEVEL_SHIFT = 0.03

class DataGen:

    def __init__(self, patient_name_list, spacing):
        self.patient_name_list = patient_name_list
        self.spacing = spacing

        print("Patients: ")
        print(self.patient_name_list)
        sdf_list = [f"./sdt2sdf_default/{p}.pkl" for p in self.patient_name_list]

        space = load_data(self.patient_name_list, sdfpkl_list=sdf_list)
        preload_all_possible(space)
        print(space.explore())
        self.space = space

    def load_batch(self):
        num_of_obj = len(self.patient_name_list)
        batch = None
        for code, patient in enumerate(self.patient_name_list):

            xyz = self.space[f'sampling_grid'](spacing=self.spacing)
            xyz_flat = xyz.reshape((-1, 3))
            sdf_flat = jnp.squeeze(self.space[f'default/{patient}/sdfpkl/repr/xyz2sdt'](xyz_flat))
            xyz_flat = jnp.array(xyz_flat)

            p_array = jnp.array(jnp.zeros((sdf_flat.shape[0], num_of_obj)))
            p_array = p_array.at[:, code].set(1.)

            DATA = ApproximateSDFLipMLP.DATA(X=xyz_flat[:, 0],
                                               Y=xyz_flat[:, 1],
                                               Z=xyz_flat[:, 2],
                                               T=jnp.zeros(sdf_flat.shape[0]),
                                               P=p_array,
                                               SDF=sdf_flat)
            print(DATA)
            if batch is None:
                batch = DATA
            else:
                batch = batch + DATA

        return batch


if __name__ == '__main__':
    # NN initialization
    task = ApproximateSDFLipMLP(batch_size=FLAT_SHAPE, model_number=5)
    rng = jax.random.PRNGKey(42)
    params, F = task.init_and_functions(rng)
    D_INIT = task.init_data()

    opt = optax.adam(LEARNING_RATE)
    opt_state = opt.init(params)

    # Batch
    name_list = ["patient009", "patient029", "patient049", "patient069", "patient089"]
    gen = DataGen(name_list, spacing=SHAPE)
    D1 = gen.load_batch()

    @jax.jit
    def train_step(params, rng, opt_state, D1):

        loss, grads = jax.value_and_grad(F.vec_main_loss)(params, rng, *tuple(D1))
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return loss, params, rng, opt_state

    max_loss = 99999
    viz = BasicVizualization(LOG_FOLDER, EXP_NAME, print_on_each_epoch=100)
    for epoch in viz.iter(EPOCHS):

        loss, params, rng, opt_state = train_step(params, rng, opt_state, D1)
        viz.record({"loss": float(loss)})

        if viz.is_print_on_epoch(epoch):
            viz.draw_loss("TRAIN_LOSS", viz._records["loss"])

            if loss < max_loss:
                predict_sdf = F.vec_sdf(params, rng, D1.X, D1.Y, D1.Z, D1.T, D1.P)
                for i, name in enumerate(name_list):
                    exact = D1.SDF[FLAT_SHAPE*i:FLAT_SHAPE*(i+1)].reshape(SHAPE)
                    predict = predict_sdf[FLAT_SHAPE*i:FLAT_SHAPE*(i+1)].reshape(SHAPE)
                    viz.sdf_to_obj(f"SDF_exact_{name}", exact, level=LEVEL_SHIFT)
                    viz.sdf_to_obj(f"SDF_predict_{name}", predict, level=LEVEL_SHIFT)
                    viz.save_3D_array(f"SDF_exact_{name}", exact)
                    viz.save_3D_array(f"SDF_predict_{name}", predict)

                viz.save_state('sdf_model', params)
                max_loss = loss

        rng, subkey = jax.random.split(rng)

    viz.draw_loss("TRAIN_LOSS", viz._records["loss"])