import jax
import optax

import nndt.space2 as spc
from nndt.trainable_task import Eikonal3D
from nndt.vizualize import BasicVizualization

LEARNING_RATE = 0.001
EPOCHS = 10001
SHAPE = (64, 64, 64)
SHAPE_FLAT = SHAPE[0] * SHAPE[1] * SHAPE[2]
EXP_NAME = 'eikonal_on_primitives'
LOG_FOLDER = f'./{EXP_NAME}/'

if __name__ == '__main__':

    space = spc.Space("main")
    space = spc.add_sphere(space, "domain", center=(0., 0., 0.), radius=2.)
    space = spc.add_sphere(space, "init_reg", center=(1., 0., 0.), radius=1.)
    print(space.print())

    domain_sdf_ = space.domain.purefun_sdf()
    init_reg_sdf_ = space.init_reg.purefun_sdf()

    task = Eikonal3D(domain_sdf_, init_reg_sdf_)

    rng_key = jax.random.PRNGKey(42)
    params, F = task.init_and_functions(rng_key)
    D_INIT = task.init_data()

    exponential_decay_scheduler = optax.exponential_decay(init_value=LEARNING_RATE,
                                                          transition_steps=EPOCHS,
                                                          decay_rate=0.70,
                                                          transition_begin=int(EPOCHS * 0.05),
                                                          staircase=False)

    opt = optax.adam(exponential_decay_scheduler)
    opt_state = opt.init(params)

    # Batch
    rng_key, subkey = jax.random.split(rng_key)
    xyz = space.domain.sampling_grid_with_noise(subkey, spacing=SHAPE, sigma=0.1).reshape((-1, 3))
    D1 = Eikonal3D.DATA(X=xyz[:, 0], Y=xyz[:, 1], Z=xyz[:, 2])

    xyz_test = space.domain.sampling_grid(spacing=SHAPE).reshape((-1, 3))
    DT = Eikonal3D.DATA(X=xyz_test[:, 0], Y=xyz_test[:, 1], Z=xyz_test[:, 2])


    @jax.jit
    def train_step(params, rng, opt_state, D1):

        loss, grads = jax.value_and_grad(F.main_loss)(params, rng, *tuple(D1))
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return loss, params, rng, opt_state


    max_loss = 99999
    viz = BasicVizualization(LOG_FOLDER, EXP_NAME, print_on_each_epoch=1000)
    for epoch in viz.iter(EPOCHS):
        loss, params, rng, opt_state = train_step(params, rng_key, opt_state, D1)
        viz.record({"loss": float(loss)})

        if epoch % 100 == 0:
            rng_key, subkey = jax.random.split(rng_key)
            xyz = space.domain.sampling_grid_with_noise(subkey, spacing=SHAPE, sigma=0.1).reshape((-1, 3))
            D1 = Eikonal3D.DATA(X=xyz[:, 0], Y=xyz[:, 1], Z=xyz[:, 2])

        if viz.is_print_on_epoch(epoch):
            solution = F.nn(params, rng, *tuple(DT)).reshape(SHAPE)
            print(solution.min(), solution.max())

            viz.save_3D_array("solution", solution)
            viz.draw_loss("TRAIN_LOSS", viz._records["loss"])

            if loss < max_loss:
                viz.save_state('sdf_model', params)
                max_loss = loss

        rng, subkey = jax.random.split(rng)
