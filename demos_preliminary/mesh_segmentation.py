import jax
import jax.numpy as jnp
import numpy as onp
import optax
from tqdm import tqdm

import nndt.space2 as spc
from nndt.space2 import split_node_test_train
from nndt.trainable_task import SurfaceSegmentation
from nndt.vizualize import BasicVizualization

LEARNING_RATE = 0.001
EPOCHS = 1601
SHAPE = (16, 16, 16)
FLAT_SHAPE = SHAPE[0] * SHAPE[1] * SHAPE[2]
EXP_NAME = 'mesh_segmentation'
LOG_FOLDER = f'./{EXP_NAME}/'


class SimpleGenerator:

    def __init__(self, rng_key, folder, spacing=(16, 16, 16),
                 test_size=0.2,
                 step=77, scale=1.,
                 sample_from_each_model=33):
        self.spacing = spacing
        self.test_size = test_size
        self.step = step
        self.scale = scale
        self.sample_from_each_model = sample_from_each_model

        self.folder = folder
        self.space = spc.load_from_path(self.folder)
        self.space.preload('shift_and_scale')
        self.space = split_node_test_train(rng_key, self.space, test_size=0.4)
        print(self.space.print())

    def __len__(self):
        return len(self.name_list)

    def get_item(self, rand_key, group='train', shift=0) -> (jnp.ndarray, jnp.ndarray):
        X = []
        Y = []
        for obj in self.space[group]:

            ns_index_center, ns_xyz_center = obj.sampling_eachN_from_mesh(count=self.sample_from_each_model,
                                                                          step=self.step,
                                                                          shift=shift)

            rgba = obj.surface_ind2rgba(ns_index_center)[:, 0:3]
            color_class = jnp.argmax(rgba, axis=1)

            for ind, xyz in enumerate(ns_xyz_center):
                _, ns_sdt = obj.surface_xyz2localsdt(xyz, spacing=self.spacing, scale=self.scale)

                ns_sdt = ns_sdt[jnp.newaxis, :, :, :, :]
                X.append(ns_sdt)
                Y.append(color_class[ind])

        X = jnp.concatenate(X, axis=0)
        Y = jnp.array(Y)

        index_array = jax.random.shuffle(rng, jnp.arange(X.shape[0], dtype=jnp.int_))
        X = jnp.take(X, index_array, axis=0)
        Y = jnp.take(Y, index_array, axis=0)

        return X, Y

    def viz_item(self, group='test', patient='patient069') -> (jnp.ndarray, jnp.ndarray):
        X = []
        Y = []
        obj = self.space[group][patient]
        num_of_points = obj.colored_obj._loader.mesh.GetNumberOfPoints()
        ns_index_center, ns_xyz_center = obj.sampling_eachN_from_mesh(count=num_of_points, step=1, shift=0)

        rgba = obj.surface_ind2rgba(ns_index_center)[:, 0:3]
        color_class = jnp.argmax(rgba, axis=1)

        for ind, xyz in tqdm(enumerate(ns_xyz_center)):
            _, ns_sdt = obj.surface_xyz2localsdt(xyz, spacing=self.spacing, scale=self.scale)
            ns_sdt = ns_sdt[jnp.newaxis, :, :, :, :]
            X.append(ns_sdt)
            Y.append(color_class[ind])

        X = jnp.concatenate(X, axis=0)
        Y = jnp.array(Y)

        save_mesh = obj.save_mesh
        return X, Y, save_mesh


if __name__ == '__main__':

    # NN initialization
    task = SurfaceSegmentation(spacing=SHAPE,
                               conv_kernel=32,
                               conv_depth=4,
                               num_classes=3,
                               batch_size=128)
    rng = jax.random.PRNGKey(42)
    params, F = task.init_and_functions(rng)

    opt = optax.adam(LEARNING_RATE)
    opt_state = opt.init(params)
    sg = SimpleGenerator(rng, '../tests/acdc_for_test')

    rng, subkey = jax.random.split(rng)
    X, Y = sg.get_item(subkey, group='train', shift=0)
    rng, subkey = jax.random.split(rng)
    test_X, test_Y = sg.get_item(subkey, group='test', shift=0)

    print(X.shape, Y.shape)
    print("Class balance: ", jnp.sum(Y == 0), jnp.sum(Y == 1), jnp.sum(Y == 2))


    @jax.jit
    def train_step(params, rng, opt_state, X, Y):
        loss, grads = jax.value_and_grad(F.main_loss)(params, rng, X, Y)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        accuracy = F.metric_accuracy(params, rng, X, Y)

        return loss, accuracy, params, rng, opt_state


    @jax.jit
    def eval_step(params, rng, X, Y):
        loss = F.main_loss(params, rng, X, Y)
        accuracy = F.metric_accuracy(params, rng, X, Y)
        return loss, accuracy


    viz_X, viz_Y, save_mesh = sg.viz_item(group='test', patient='patient069')

    max_loss = 99999
    viz = BasicVizualization(LOG_FOLDER, EXP_NAME, print_on_each_epoch=20)
    for epoch in viz.iter(EPOCHS):

        loss, accuracy, params, rng, opt_state = train_step(params, rng, opt_state, X, Y)
        val_loss, val_accuracy = eval_step(params, rng, test_X, test_Y)

        viz.record({"loss": float(loss), "accuracy": float(accuracy),
                    "val_loss": float(val_loss), "val_accuracy": float(val_accuracy)})

        if viz.is_print_on_epoch(epoch):
            viz.draw_loss("TRAIN_LOSS", viz._records["loss"])
            viz.draw_loss("TRAIN_ACCURACY", viz._records["accuracy"])
            viz.draw_loss("TEST_LOSS", viz._records["val_loss"])
            viz.draw_loss("TEST_ACCURACY", viz._records["val_accuracy"])
            rng, subkey = jax.random.split(rng)

            viz_Y_pred = F.nn(params, rng, viz_X)
            viz_Y_pred = jnp.argmax(viz_Y_pred, -1)
            viz.save_mesh('patient069', save_mesh, {"pred_class": onp.array(viz_Y_pred), "class": onp.array(viz_Y)})

            X, Y = sg.get_item(subkey, group='train', shift=4 * int(epoch / viz.print_on_each_epoch))
            print("Class balance: ", jnp.sum(Y == 0), jnp.sum(Y == 1), jnp.sum(Y == 2))
