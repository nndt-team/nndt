import jax
import jax.numpy as jnp
import numpy as onp
import optax
import os
from tqdm import tqdm

from nndt.space.loaders import load_data, preload_all_possible, Object
from nndt.trainable_task import SurfaceSegmentation
from nndt.vizualize import BasicVizualization

LEARNING_RATE = 0.001
EPOCHS = 1601
SHAPE = (16, 16, 16)
FLAT_SHAPE = SHAPE[0] * SHAPE[1] * SHAPE[2]
EXP_NAME = 'mesh_segmentation'
LOG_FOLDER = f'./{EXP_NAME}/'


class SimpleGenerator:

    def __init__(self, folder, spacing=(16, 16, 16),
                 test_size=0.2,
                 step=77, scale=1.,
                 sample_from_each_model=33):
        self.spacing = spacing
        self.test_size = test_size
        self.step = step
        self.scale = scale
        self.sample_from_each_model = sample_from_each_model

        self.name_list = os.listdir(folder)
        self.name_list.sort()
        self.mesh_list = [f"{folder}/{p}/colored.obj" for p in self.name_list]
        self.sdt_list = [f"{folder}/{p}/sdf.npy" for p in self.name_list]
        self.space = load_data(self.name_list, self.mesh_list, self.sdt_list, test_size=test_size)
        preload_all_possible(self.space)
        print(self.space.explore())

    def __len__(self):
        return len(self.name_list)

    def get_item(self, rand_key, group='train', shift=0) -> (jnp.ndarray, jnp.ndarray):
        X = []
        Y = []
        for object in self.space[group]:
            if isinstance(object, Object):

                ns_index_center, ns_xyz_center = object[f'mesh/repr/sampling_eachN'](count=self.sample_from_each_model,
                                                                                     step=self.step, shift=shift)

                red = object[f'mesh/repr/point_color'].red[ns_index_center]
                green = object[f'mesh/repr/point_color'].green[ns_index_center]
                blue = object[f'mesh/repr/point_color'].blue[ns_index_center]
                color_class = jnp.argmax(jnp.array([red, green, blue]), axis=0)

                for ind, xyz in enumerate(ns_xyz_center):
                    _, ns_sdt = object[f'sdt/repr/xyz2local_sdt'](xyz, spacing=self.spacing, scale=self.scale)

                    ns_sdt = ns_sdt[jnp.newaxis, :, :, :, :]
                    X.append(ns_sdt)
                    Y.append(color_class[ind])

        X = jnp.concatenate(X, axis=0)
        Y = jnp.array(Y)

        index_array = jax.random.shuffle(rng, jnp.arange(X.shape[0], dtype=jnp.int_))
        X = jnp.take(X, index_array, axis=0)
        Y = jnp.take(Y, index_array, axis=0)

        return X, Y

    def viz_item(self, group='test', patient='patient029') -> (jnp.ndarray, jnp.ndarray):
        X = []
        Y = []
        object = self.space[group][patient]
        num_of_points = object["mesh/repr"].surface_mesh2.mesh.GetNumberOfPoints()
        ns_index_center, ns_xyz_center = object[f'mesh/repr/sampling_eachN'](count=num_of_points, step=1, shift=0)

        red = object[f'mesh/repr/point_color'].red[ns_index_center]
        green = object[f'mesh/repr/point_color'].green[ns_index_center]
        blue = object[f'mesh/repr/point_color'].blue[ns_index_center]
        color_class = jnp.argmax(jnp.array([red, green, blue]), axis=0)

        for ind, xyz in tqdm(enumerate(ns_xyz_center)):
            _, ns_sdt = object[f'sdt/repr/xyz2local_sdt'](xyz, spacing=self.spacing, scale=self.scale)
            ns_sdt = ns_sdt[jnp.newaxis, :, :, :, :]
            X.append(ns_sdt)
            Y.append(color_class[ind])

        X = jnp.concatenate(X, axis=0)
        Y = jnp.array(Y)

        save_mesh = object[f'mesh/repr/save_mesh']
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
    sg = SimpleGenerator('../tests/acdc_for_test')

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


    viz_X, viz_Y, save_mesh = sg.viz_item(group='test', patient='patient029')

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
            viz.save_mesh('patient029', save_mesh, {"pred_class": onp.array(viz_Y_pred), "class": onp.array(viz_Y)})

            X, Y = sg.get_item(subkey, group='train', shift=4 * int(epoch / viz.print_on_each_epoch))
            print("Class balance: ", jnp.sum(Y == 0), jnp.sum(Y == 1), jnp.sum(Y == 2))
