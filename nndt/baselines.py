import os.path
import pickle

import jax
import jax.numpy as jnp
import optax

from nndt.math_core import grid_in_cube2
from nndt.space.loaders import load_data, preload_all_possible
from nndt.trainable_task import SimpleSDF
from nndt.vizualize import BasicVizualization


class Sdt2sdf:

    def __init__(self,
                 file_sdf,
                 file_model,
                 file_params,
                 spacing=(64, 64, 64),
                 learning_rate=0.006,
                 epochs=10001,
                 name='default_model'):
        assert (os.path.exists(self.sdf_file))

        self.name = name
        self.file_sdf = file_sdf
        self.file_model = file_model
        self.file_params = file_params
        self.spacing = spacing
        self.flat_shape = spacing[0]*spacing[1]*spacing[2]
        self.learning_rate = learning_rate
        self.epochs = epochs


        self.space = load_data([self.name], None, [self.file_sdf])
        preload_all_possible(self.space)

    def load_batch(self):

        print(self.space.explore())

        code, patient = 0, self.name

        xyz = self.space[f'sampling_grid'](spacing=self.spacing)
        xyz_flat = xyz.reshape((-1, 3))
        sdf_flat = jnp.squeeze(self.space[f'default/{patient}/sdt/repr/xyz2sdt'](xyz_flat))
        xyz_flat = jnp.array(xyz_flat)

        DATA = SimpleSDF.DATA(X=xyz_flat[:, 0], Y=xyz_flat[:, 1], Z=xyz_flat[:, 2], SDF=sdf_flat)

        return DATA


    def convert(self):

        kwargs = {"mlp_layers" : (32, 32, 32, 32, 32, 32, 32, 32, 1),
                 "batch_size" : 64*64*64}
        # NN initialization
        task = SimpleSDF(**kwargs)
        rng = jax.random.PRNGKey(42)
        params, F = task.init_and_functions(rng)

        opt = optax.adam(self.learning_rate)
        opt_state = opt.init(params)

        D1 = self.load_batch()

        @jax.jit
        def train_step(params, rng, opt_state):
            loss, grads = jax.value_and_grad(F.vec_main_loss)(params, rng, *tuple(D1))
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return loss, params, rng, opt_state

        min_loss = 99999999
        loss_history = []
        for epoch in range(self.epochs):

            loss, params, rng, opt_state = train_step(params, rng, opt_state, D1)
            loss_history.append(float(loss))

            if loss < min_loss:
                pickle.dump(task, open(self.file_model, 'wb'))
                pickle.dump(params, open(self.file_params, 'wb'))
                min_loss = loss