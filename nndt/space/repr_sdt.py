import pickle
import warnings

import jax
import jax.numpy as jnp
import optax
from packaging import version
from tqdm import tqdm

import nndt
from nndt.math_core import grid_in_cube
from nndt.space.abstracts import *
from nndt.space.sources import SDFPKLSource, SDTSource
from nndt.space.utils import downup_update_bbox
from nndt.space.vtk_wrappers import *
from nndt.trainable_task import SimpleSDF


class AbstractSDXRepr(AbstractRegion, ExtendedNodeMixin, UnloadMixin):
    @abstractmethod
    def ps_xyz2sdt(self, ps_xyz: onp.ndarray) -> onp.ndarray:
        pass

    @abstractmethod
    def ns_xyz2sdt(self, ns_xyz: onp.ndarray) -> onp.ndarray:
        pass


class SDTRepr(AbstractSDXRepr):
    MAGIC_CORRECTION = 0.503  # This is absolutely magic coefficient that reduce error between bboxes to 0.49075 mm

    def __init__(
        self,
        parent: FileSource,
        sdt_explicit_array2: SDTExplicitArray,
        physical_center: (float, float, float),
        physical_bbox: ((float, float, float), (float, float, float)),
        normed_center: (float, float, float),
        normed_bbox: ((float, float, float), (float, float, float)),
        scale_physical2normed: float,
        _ndim=3,
        _scale=1.0,
        name="",
    ):
        super(SDTRepr, self).__init__(_ndim=_ndim, _bbox=normed_bbox, name=name)
        self.name = name
        self.parent = parent
        self._sdt_explicit_array2 = sdt_explicit_array2

        self.physical_center = onp.array(physical_center)
        self.physical_bbox = physical_bbox
        self.normed_center = onp.array(normed_center)
        self.normed_bbox = normed_bbox

        self.scale_physical2normed = scale_physical2normed
        self._print_color = Fore.GREEN

    def params_to_json(self):
        return {
            "physical_center": self.physical_center,
            "physical_bbox": self.physical_bbox,
            "normed_center": self.normed_center,
            "normed_bbox": self.normed_bbox,
            "scale_physical2normed": self.scale_physical2normed,
        }

    def unload_data(self):
        self._sdt_explicit_array2.unload_data()

    def is_data_load(self):
        return self._sdt_explicit_array2.is_data_load()

    def ps_xyz2sdt(self, ps_xyz: onp.ndarray) -> onp.ndarray:
        ps_sdt = self._sdt_explicit_array2.request(ps_xyz)
        return ps_sdt

    def ns_xyz2sdt(self, ns_xyz: onp.ndarray) -> onp.ndarray:
        ps_xyz = (
            ns_xyz - self.normed_center
        ) * self.scale_physical2normed + self.physical_center
        ps_sdt = self._sdt_explicit_array2.request(ps_xyz)
        ns_sdt = ps_sdt / self.scale_physical2normed
        return ns_sdt

    @classmethod
    def load_mesh_and_bring_to_center(
        cls, source: SDTSource, padding_physical=(10, 10, 10), scale_physical2normed=50
    ):
        sdt_explicit_array2 = SDTExplicitArray(source.filepath)
        Xmin, Xmax, Ymin, Ymax, Zmin, Zmax = sdt_explicit_array2.min_bbox()
        sdt_explicit_array2.unload_data()

        normed_center = (0.0, 0.0, 0.0)
        physical_bbox = (
            (
                Xmin - padding_physical[0] + SDTRepr.MAGIC_CORRECTION,
                Ymin - padding_physical[1] + SDTRepr.MAGIC_CORRECTION,
                Zmin - padding_physical[2] + SDTRepr.MAGIC_CORRECTION,
            ),
            (
                Xmax + padding_physical[0] - SDTRepr.MAGIC_CORRECTION,
                Ymax + padding_physical[1] - SDTRepr.MAGIC_CORRECTION,
                Zmax + padding_physical[2] - SDTRepr.MAGIC_CORRECTION,
            ),
        )
        physical_center = (
            physical_bbox[0][0] + (physical_bbox[1][0] - physical_bbox[0][0]) / 2.0,
            physical_bbox[0][1] + (physical_bbox[1][1] - physical_bbox[0][1]) / 2.0,
            physical_bbox[0][2] + (physical_bbox[1][2] - physical_bbox[0][2]) / 2.0,
        )

        scale_physical2normed = scale_physical2normed

        normed_bbox = (
            (
                (physical_bbox[0][0] - physical_center[0]) / scale_physical2normed
                + normed_center[0],
                (physical_bbox[0][1] - physical_center[1]) / scale_physical2normed
                + normed_center[1],
                (physical_bbox[0][2] - physical_center[2]) / scale_physical2normed
                + normed_center[2],
            ),
            (
                (physical_bbox[1][0] - physical_center[0]) / scale_physical2normed
                + normed_center[0],
                (physical_bbox[1][1] - physical_center[1]) / scale_physical2normed
                + normed_center[1],
                (physical_bbox[1][2] - physical_center[2]) / scale_physical2normed
                + normed_center[2],
            ),
        )

        repr = SDTRepr(
            source,
            sdt_explicit_array2,
            physical_center,
            physical_bbox,
            normed_center,
            normed_bbox,
            scale_physical2normed,
            name="repr",
        )

        downup_update_bbox(repr)

        return repr


class Xyz2SDT(AbstractMethod, ExtendedNodeMixin):
    def __init__(self, parent: AbstractSDXRepr):
        super(Xyz2SDT, self).__init__()
        self.name = "xyz2sdt"
        self.parent = parent

    def __repr__(self):
        return f"xyz2sdt(ns_xyz[...,3]) -> ns_sdt[...,1]"

    def __call__(self, ns_xyz: jnp.ndarray) -> jnp.ndarray:
        ns_sdt = self.parent.ns_xyz2sdt(ns_xyz)
        return ns_sdt


class Xyz2LocalSDT(AbstractMethod, ExtendedNodeMixin):
    def __init__(self, parent: AbstractSDXRepr):
        super(Xyz2LocalSDT, self).__init__()
        self.name = "xyz2local_sdt"
        self.parent = parent

    def __repr__(self):
        return f"xyz2local_sdt(ns_xyz[3], spacing=(D,H,W), scale=1.) -> ns_xyz[D,H,W,3], ns_local_sdt[D,H,W,1]"

    def __call__(
        self, ns_xyz: (float, float, float), spacing=(2, 2, 2), scale=1.0
    ) -> (jnp.ndarray, jnp.ndarray):
        ns_xyz = jnp.array(ns_xyz)
        ns_cube = grid_in_cube(
            spacing=spacing, scale=scale, center_shift=(0.0, 0.0, 0.0)
        )
        ns_cube = ns_cube + ns_xyz
        ns_cube = ns_cube.reshape((-1, 3))
        ns_local_sdt = self.parent.ns_xyz2sdt(ns_cube)
        ns_local_sdt = ns_local_sdt.reshape(spacing)[:, :, :, jnp.newaxis]
        ns_cube = ns_cube.reshape((spacing[0], spacing[1], spacing[2], 3))
        return ns_cube, ns_local_sdt


class TrainSDT2SDF(AbstractMethod, ExtendedNodeMixin):
    def __init__(self, parent: SDTRepr):
        super(TrainSDT2SDF, self).__init__()
        self.name = "train_sdt2sdf"
        self.parent = parent

    def __repr__(self):
        return f"train_sdt2sdf(filename, spacing=(64, 64, 64), width=32, depth=8, learning_rate=0.006, epochs=10001)"

    def load_batch(self, spacing):
        xyz = self.parent[f"sampling_grid"](spacing=spacing)
        xyz_flat = xyz.reshape((-1, 3))
        sdf_flat = jnp.squeeze(self.parent[f"xyz2sdt"](xyz_flat))
        xyz_flat = jnp.array(xyz_flat)

        data = SimpleSDF.DATA(
            X=xyz_flat[:, 0], Y=xyz_flat[:, 1], Z=xyz_flat[:, 2], SDF=sdf_flat
        )
        return data

    def __call__(
        self,
        filename,
        spacing=(64, 64, 64),
        width=32,
        depth=8,
        learning_rate=0.006,
        epochs=10001,
    ):

        kwargs = {
            "mlp_layers": tuple([width] * depth + [1]),
            "batch_size": spacing[0] * spacing[1] * spacing[2],
        }

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
        for epoch in (pbar := tqdm(range(epochs))):

            loss, params, rng, opt_state = train_step(params, rng, opt_state, D1)
            loss_history.append(float(loss))
            pbar.set_description(f"min_loss = {min_loss:.06f}")

            if loss < min_loss:
                pickle.dump(
                    {
                        "version": nndt.__version__,
                        "repr": self.parent.params_to_json(),
                        "trainable_task": kwargs,
                        "history_loss": loss_history,
                        "params": params,
                    },
                    open(filename + ".pkl", "wb"),
                )
                min_loss = loss


class SDFRepr(AbstractSDXRepr):
    MAGIC_CORRECTION = 0.503  # This is absolutely magic coefficient that reduce error between bboxes to 0.49075 mm

    def __init__(
        self,
        parent: SDFPKLSource,
        trainable_task: SimpleSDF,
        trainable_params,
        physical_center: (float, float, float),
        physical_bbox: ((float, float, float), (float, float, float)),
        normed_center: (float, float, float),
        normed_bbox: ((float, float, float), (float, float, float)),
        scale_physical2normed: float,
        _ndim=3,
        _scale=1.0,
        name="repr",
    ):
        super(SDFRepr, self).__init__(_ndim=_ndim, _bbox=normed_bbox, name=name)
        self.name = name
        self.parent = parent

        self.trainable_task = trainable_task
        self.trainable_params = trainable_params

        self.physical_center = onp.array(physical_center)
        self.physical_bbox = physical_bbox
        self.normed_center = onp.array(normed_center)
        self.normed_bbox = normed_bbox

        self.scale_physical2normed = scale_physical2normed
        self._print_color = Fore.GREEN

        rng = jax.random.PRNGKey(42)
        _, self.F = self.trainable_task.init_and_functions(rng)

    def ps_xyz2sdt(self, ps_xyz: onp.ndarray) -> onp.ndarray:
        ns_xyz = (
            ps_xyz - self.physical_center
        ) / self.scale_physical2normed + self.normed_center
        ns_sdf = self.ns_xyz2sdt(ns_xyz)
        ps_sdt = ns_sdf * self.scale_physical2normed

        return ps_sdt

    def ns_xyz2sdt(self, ns_xyz: onp.ndarray) -> onp.ndarray:
        ret_shape = list(ns_xyz.shape)
        ret_shape[-1] = 1
        ret_shape = tuple(ret_shape)

        ns_xyz_flat = ns_xyz.reshape((-1, 3))
        rng = jax.random.PRNGKey(42)
        ns_sdf = self.F.vec_sdf(
            self.trainable_params,
            rng,
            ns_xyz_flat[:, 0],
            ns_xyz_flat[:, 1],
            ns_xyz_flat[:, 2],
        )

        ns_sdf = ns_sdf.reshape(ret_shape)
        return ns_sdf

    def unload_data(self):
        warnings.warn("Data unloading is not implemented yet for SDFRepr!")
        pass

    def is_data_load(self) -> bool:
        return True

    @classmethod
    def load_from_json(cls, source: SDFPKLSource):
        with open(source.filepath, "rb") as input_file:
            result = pickle.load(input_file)

            version_ = result["version"]
            repr_ = result["repr"]
            trainable_task_ = result["trainable_task"]
            history_loss_ = result["history_loss"]
            params_ = result["params"]

        if version.parse(nndt.__version__) < version.parse(version_):
            warnings.warn(
                "Loaded neural network was created on earlier version of NNDT!"
            )

        task = SimpleSDF(**trainable_task_)

        sdf_repr = SDFRepr(
            parent=source,
            trainable_task=task,
            trainable_params=params_,
            physical_center=repr_["physical_center"],
            physical_bbox=repr_["physical_bbox"],
            normed_center=repr_["normed_center"],
            normed_bbox=repr_["normed_bbox"],
            scale_physical2normed=repr_["scale_physical2normed"],
        )

        downup_update_bbox(sdf_repr)

        return sdf_repr
