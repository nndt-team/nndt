import os

import jax
import jax.numpy as jnp
import optax

from nndt.math_core import grid_in_cube2
from nndt.space.loaders import load_data, preload_all_possible
from nndt.trainable_task import ApproximateSDF
from nndt.vizualize import BasicVizualization

LEARNING_RATE = 0.006
EPOCHS = 9001
SHAPE = (64, 64, 64)
FLAT_SHAPE = SHAPE[0] * SHAPE[1] * SHAPE[2]
EXP_NAME = 'sdf_multiple_files'
LOG_FOLDER = f'./{EXP_NAME}/'
LEVEL_SHIFT = 0.03


def load_batch(patient_name_list, spacing=(2, 2, 2)):
    patient_name_list = patient_name_list
    num_of_obj = len(patient_name_list)
    print("Patients: ")
    print(patient_name_list)
    mesh_list = [f"../tests/acdc_for_test/{p}/colored.obj" for p in patient_name_list]
    sdt_list = [f"../tests/acdc_for_test/{p}/sdf.npy" for p in patient_name_list]

    space = load_data(patient_name_list, mesh_list, sdt_list)
    preload_all_possible(space)
    print(space.explore())

    batch = None
    for code, patient in enumerate(patient_name_list):

        xyz = space[f'sampling_grid'](spacing=spacing)
        xyz_flat = xyz.reshape((-1, 3))
        sdf_flat = jnp.squeeze(space[f'default/{patient}/sdt/repr/xyz2sdt'](xyz_flat))
        xyz_flat = jnp.array(xyz_flat)

        p_array = jnp.array(jnp.zeros((sdf_flat.shape[0], num_of_obj)))
        p_array = p_array.at[:, code].set(1.)

        DATA = ApproximateSDF.DATA(X=xyz_flat[:, 0],
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
    task = ApproximateSDF(batch_size=FLAT_SHAPE, model_number=2)
    rng = jax.random.PRNGKey(42)
    params, F = task.init_and_functions(rng)
    D_INIT = task.init_data()

    opt = optax.adam(LEARNING_RATE)
    opt_state = opt.init(params)

    # Batch
    D1 = load_batch(["patient009", "patient029"], spacing=SHAPE)

    a_test = jnp.full((FLAT_SHAPE, 1), 0.6666666666)
    b_test = jnp.full((FLAT_SHAPE, 1), 0.3333333333)
    part0_test = jnp.concatenate([a_test, b_test], axis=1)
    part1_test = jnp.concatenate([b_test, a_test], axis=1)

    xyz = grid_in_cube2(spacing=SHAPE,
                        lower=jnp.array((D1.X.min(), D1.Y.min(), D1.Z.min())),
                        upper=jnp.array((D1.X.max(), D1.Y.max(), D1.Z.max()))).reshape((-1, 3))
    D_TEST = ApproximateSDF.DATA(X=jnp.hstack([xyz[:, 0], xyz[:, 0]]),
                                 Y=jnp.hstack([xyz[:, 1], xyz[:, 1]]),
                                 Z=jnp.hstack([xyz[:, 2], xyz[:, 2]]),
                                 T=jnp.zeros(FLAT_SHAPE * 2),
                                 P=jnp.concatenate([part0_test, part1_test], axis=0),
                                 SDF=None)


    @jax.jit
    def train_step(params, rng, opt_state):

        loss, grads = jax.value_and_grad(F.vec_main_loss)(params, rng, *tuple(D1))
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return loss, params, rng, opt_state


    max_loss = 99999
    viz = BasicVizualization(LOG_FOLDER, EXP_NAME, print_on_each_epoch=1000)
    for epoch in viz.iter(EPOCHS):

        loss, params, rng, opt_state = train_step(params, rng, opt_state)

        viz.record({"loss": float(loss)})

        if viz.is_print_on_epoch(epoch):
            tmp = F.vec_sdf(params, rng,
                            D1.X, D1.Y, D1.Z,
                            D1.T, D1.P)
            model0000 = tmp[:FLAT_SHAPE].reshape(SHAPE)
            model1000 = tmp[FLAT_SHAPE:].reshape(SHAPE)

            tmp1 = F.vec_sdf(params, rng,
                             D_TEST.X, D_TEST.Y, D_TEST.Z,
                             D_TEST.T, D_TEST.P)

            model0333 = tmp1[:FLAT_SHAPE].reshape(SHAPE)
            model0666 = tmp1[FLAT_SHAPE:].reshape(SHAPE)

            sdf_val0 = D1.SDF[:FLAT_SHAPE].reshape(SHAPE)
            sdf_val1 = D1.SDF[FLAT_SHAPE:].reshape(SHAPE)

            viz.sdf_to_obj("SDF_0000_exact", sdf_val0)
            viz.sdf_to_obj("SDF_0000", model0000, level=LEVEL_SHIFT)
            viz.sdf_to_obj("SDF_0333", model0333, level=LEVEL_SHIFT)
            viz.sdf_to_obj("SDF_0666", model0666, level=LEVEL_SHIFT)
            viz.sdf_to_obj("SDF_1000", model1000, level=LEVEL_SHIFT)
            viz.sdf_to_obj("SDF_1000_exact", sdf_val1)

            viz.save_3D_array("SDF_0000", model0000)
            viz.save_3D_array("SDF_0333", model0333)
            viz.save_3D_array("SDF_0666", model0666)
            viz.save_3D_array("SDF_1000", model1000)

            viz.draw_loss("TRAIN_LOSS", viz._records["loss"])

            if loss < max_loss:
                viz.save_state('sdf_model', params)
                max_loss = loss

        rng, subkey = jax.random.split(rng)
