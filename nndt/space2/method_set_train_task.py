import pickle
from typing import Union

import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

import nndt
from nndt import SimpleSDF
from nndt.space2.abstracts import AbstractBBoxNode, node_method
from nndt.space2.filesource import FileSource
from nndt.space2.implicit_representation import ImpRepr
from nndt.space2.method_set import MethodSetNode
from nndt.space2.transformation import AbstractTransformation


class TrainTaskSetNode(MethodSetNode):
    def __init__(
        self,
        object_3d: AbstractBBoxNode,
        sdt: Union[FileSource, ImpRepr],
        transform: AbstractTransformation,
        parent: AbstractBBoxNode = None,
    ):
        super(TrainTaskSetNode, self).__init__("train_task", parent=parent)
        self.object_3d = object_3d
        assert isinstance(sdt, ImpRepr) or sdt.loader_type == "sdt"
        self.sdt = sdt
        self.transform = transform

    def load_batch(self, spacing):
        # TODO! This place is a dangerous. Sampling initializes after this class.
        # This call does not specialize specific sampling node or sdt!
        # I hope I will find a proper way to write this in future.
        xyz = self.parent.sampling.sampling_grid(spacing=spacing)
        xyz_flat = xyz.reshape((-1, 3))
        sdf_flat = jnp.squeeze(self.parent.sdt.surface_xyz2sdt(xyz_flat))
        xyz_flat = jnp.array(xyz_flat)

        data = SimpleSDF.DATA(
            X=xyz_flat[:, 0], Y=xyz_flat[:, 1], Z=xyz_flat[:, 2], SDF=sdf_flat
        )
        return data

    @node_method("train_task_sdt2sdf(filename, **kwargs)")
    def train_task_sdt2sdf(
        self,
        filename,
        spacing=(64, 64, 64),
        width=32,
        depth=8,
        learning_rate=0.006,
        epochs=10001,
    ):
        if not (
            hasattr(self.parent, "sampling")
            and hasattr(self.parent.sampling, "sampling_grid")
        ):
            raise NotImplementedError(
                "This error is really bad. Initialization order was broken!"
            )
        if not (
            hasattr(self.parent, "sdt") and hasattr(self.parent.sdt, "surface_xyz2sdt")
        ):
            raise NotImplementedError(
                "This error is really bad. Initialization order was broken!"
            )

        kwargs = {
            "mlp_layers": tuple([width] * depth + [1]),
            "batch_size": spacing[0] * spacing[1] * spacing[2],
        }

        from nndt.trainable_task import SimpleSDF

        task = SimpleSDF(**kwargs)
        rng = jax.random.PRNGKey(42)
        params, F = task.init_and_functions(rng)

        opt = optax.adam(learning_rate)
        opt_state = opt.init(params)

        D1 = self.load_batch(spacing)

        @jax.jit
        def train_step(params, rng, opt_state, D1):
            loss, grads = jax.value_and_grad(F.vec_main_loss)(params, rng, *tuple(D1))
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return loss, params, rng, opt_state

        min_loss = 99999999
        loss_history = []
        pbar = tqdm(range(epochs))
        for epoch in pbar:

            loss, params, rng, opt_state = train_step(params, rng, opt_state, D1)
            loss_history.append(float(loss))
            pbar.set_description(f"min_loss = {min_loss:.06f}")

            if loss < min_loss:
                with open(filename, "wb") as fl:
                    pickle.dump(
                        {
                            "version": nndt.__version__,
                            "repr": {
                                (k, v)
                                for k, v in self.transform.__dict__.items()
                                if isinstance(v, (int, float, str))
                            },
                            "bbox": self.object_3d.bbox,
                            "trainable_task": kwargs,
                            "history_loss": loss_history,
                            "params": params,
                        },
                        fl,
                    )
                min_loss = loss
