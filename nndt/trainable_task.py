from abc import abstractmethod
from collections import namedtuple
from typing import Callable, NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
from jax.random import KeyArray

from nndt.haiku_modules import DescConv, LipMLP


class AbstractTrainableTask:
    @abstractmethod
    def init_data(self) -> namedtuple:
        pass

    @abstractmethod
    def init_and_functions(self, rng_key: KeyArray) -> (namedtuple, namedtuple):
        pass


class SimpleSDF(AbstractTrainableTask):
    """
    This is a trainable task for the problem of shape interpolation for one 3d object.
    This class employs the usual multi-layer perceptron.
    """

    class FUNC(NamedTuple):
        sdf: Callable
        vec_sdf: Callable
        sdf_dx: Callable
        sdf_dy: Callable
        sdf_dz: Callable
        vec_sdf_dx: Callable
        vec_sdf_dy: Callable
        vec_sdf_dz: Callable
        vec_main_loss: Callable

    class DATA(NamedTuple):
        X: jnp.ndarray  # [N]
        Y: jnp.ndarray  # [N]
        Z: jnp.ndarray  # [N]
        SDF: jnp.ndarray  # [N]

        def __add__(self, other):
            return SimpleSDF.DATA(
                X=jnp.concatenate([self.X, other.X], axis=0),
                Y=jnp.concatenate([self.Y, other.Y], axis=0),
                Z=jnp.concatenate([self.Z, other.Z], axis=0),
                SDF=jnp.concatenate([self.SDF, other.SDF], axis=0),
            )

    def __init__(
        self, mlp_layers=(32, 32, 32, 32, 32, 32, 32, 32, 1), batch_size=64 * 64 * 64
    ):
        self.mlp_layers = mlp_layers
        self.batch_size = batch_size
        self._init_data = SimpleSDF.DATA(
            X=jnp.zeros(self.batch_size),
            Y=jnp.zeros(self.batch_size),
            Z=jnp.zeros(self.batch_size),
            SDF=jnp.zeros(self.batch_size),
        )

    def init_and_functions(self, rng_key: KeyArray) -> (namedtuple, namedtuple):
        def constructor():
            def f_sdf(x, y, z):
                vec = jnp.hstack([x, y, z])
                net = hk.nets.MLP(output_sizes=self.mlp_layers, activation=jnp.tanh)
                return jnp.squeeze(net(vec))

            vec_f_sdf = hk.vmap(f_sdf, in_axes=(0, 0, 0), split_rng=False)

            def vec_main_loss(X, Y, Z, SDF):
                return jnp.mean((vec_f_sdf(X, Y, Z) - SDF) ** 2)

            grad_x = hk.grad(f_sdf, argnums=0)
            grad_y = hk.grad(f_sdf, argnums=1)
            grad_z = hk.grad(f_sdf, argnums=2)

            vec_grad_x = hk.vmap(grad_x, in_axes=(0, 0, 0), split_rng=False)
            vec_grad_y = hk.vmap(grad_y, in_axes=(0, 0, 0), split_rng=False)
            vec_grad_z = hk.vmap(grad_z, in_axes=(0, 0, 0), split_rng=False)

            def init(X, Y, Z, SDF):
                return vec_main_loss(X, Y, Z, SDF)

            return init, SimpleSDF.FUNC(
                sdf=f_sdf,
                vec_sdf=vec_f_sdf,
                sdf_dx=grad_x,
                sdf_dy=grad_y,
                sdf_dz=grad_z,
                vec_sdf_dx=vec_grad_x,
                vec_sdf_dy=vec_grad_y,
                vec_sdf_dz=vec_grad_z,
                vec_main_loss=vec_main_loss,
            )

        init, function_tuple = hk.multi_transform(constructor)
        functions = SimpleSDF.FUNC(*function_tuple)
        init_params = init(rng_key, *tuple(self._init_data))

        return init_params, functions


class ApproximateSDF(AbstractTrainableTask):
    """
    This is a trainable task for the problem of shape interpolation between several 3d objects.
    This class employs the usual multi-layer perceptron.
    """

    FUNC = namedtuple(
        "ApproximateSDF_DATA",
        [
            "sdf",
            "vec_sdf",
            "sdf_dx",
            "sdf_dy",
            "sdf_dz",
            "sdf_dt",
            "vec_sdf_dx",
            "vec_sdf_dy",
            "vec_sdf_dz",
            "vec_sdf_dt",
            "vec_main_loss",
        ],
    )

    class DATA(NamedTuple):
        X: jnp.ndarray  # [N]
        Y: jnp.ndarray  # [N]
        Z: jnp.ndarray  # [N]
        T: jnp.ndarray  # [N]
        P: jnp.ndarray  # [N]
        SDF: jnp.ndarray  # [N]

        def __add__(self, other):
            return ApproximateSDF.DATA(
                X=jnp.concatenate([self.X, other.X], axis=0),
                Y=jnp.concatenate([self.Y, other.Y], axis=0),
                Z=jnp.concatenate([self.Z, other.Z], axis=0),
                T=jnp.concatenate([self.T, other.T], axis=0),
                P=jnp.concatenate([self.P, other.P], axis=0),
                SDF=jnp.concatenate([self.SDF, other.SDF], axis=0),
            )

    def __init__(
        self,
        mlp_layers=(64, 64, 64, 64, 64, 64, 64, 64, 1),
        batch_size=262144,
        model_number=2,
    ):
        self.mlp_layers = mlp_layers
        self.batch_size = batch_size
        self.model_number = model_number

        self._init_data = ApproximateSDF.DATA(
            X=jnp.zeros(self.batch_size),
            Y=jnp.zeros(self.batch_size),
            Z=jnp.zeros(self.batch_size),
            T=jnp.zeros(self.batch_size),
            P=jnp.zeros((self.batch_size, self.model_number)),
            SDF=jnp.zeros(self.batch_size),
        )

    def init_data(self) -> namedtuple:
        return self._init_data

    def init_and_functions(self, rng_key: KeyArray) -> (namedtuple, namedtuple):
        def constructor():
            def f_sdf(x, y, z, t, p):
                vec = jnp.hstack([x, y, z, t, p])
                net = hk.nets.MLP(output_sizes=self.mlp_layers, activation=jnp.tanh)
                return jnp.squeeze(net(vec))

            vec_f_sdf = hk.vmap(f_sdf, in_axes=(0, 0, 0, 0, 0), split_rng=False)

            def vec_main_loss(X, Y, Z, T, P, SDF):
                return jnp.mean((vec_f_sdf(X, Y, Z, T, P) - SDF) ** 2)

            grad_x = hk.grad(f_sdf, argnums=0)
            grad_y = hk.grad(f_sdf, argnums=1)
            grad_z = hk.grad(f_sdf, argnums=2)
            grad_t = hk.grad(f_sdf, argnums=3)

            vec_grad_x = hk.vmap(grad_x, in_axes=(0, 0, 0, 0, 0), split_rng=False)
            vec_grad_y = hk.vmap(grad_y, in_axes=(0, 0, 0, 0, 0), split_rng=False)
            vec_grad_z = hk.vmap(grad_z, in_axes=(0, 0, 0, 0, 0), split_rng=False)
            vec_grad_t = hk.vmap(grad_t, in_axes=(0, 0, 0, 0, 0), split_rng=False)

            def init(X, Y, Z, T, P, SDF):
                return vec_main_loss(X, Y, Z, T, P, SDF)

            return init, ApproximateSDF.FUNC(
                sdf=f_sdf,
                vec_sdf=vec_f_sdf,
                sdf_dx=grad_x,
                sdf_dy=grad_y,
                sdf_dz=grad_z,
                sdf_dt=grad_t,
                vec_sdf_dx=vec_grad_x,
                vec_sdf_dy=vec_grad_y,
                vec_sdf_dz=vec_grad_z,
                vec_sdf_dt=vec_grad_t,
                vec_main_loss=vec_main_loss,
            )

        init, function_tuple = hk.multi_transform(constructor)
        functions = ApproximateSDF.FUNC(*function_tuple)
        init_params = init(rng_key, *tuple(self._init_data))

        return init_params, functions


class ApproximateSDFLipMLP(AbstractTrainableTask):
    """
    This is a trainable task for the problem of shape interpolation between several 3d objects.
    This class employs a multi-layer perceptron with Lipschitz regularization (LipMLP).
    """

    FUNC = namedtuple(
        "ApproximateSDFLipMLP_DATA",
        [
            "sdf",
            "vec_sdf",
            "sdf_dx",
            "sdf_dy",
            "sdf_dz",
            "sdf_dt",
            "vec_sdf_dx",
            "vec_sdf_dy",
            "vec_sdf_dz",
            "vec_sdf_dt",
            "vec_main_loss",
        ],
    )

    class DATA(NamedTuple):
        X: jnp.ndarray  # [N]
        Y: jnp.ndarray  # [N]
        Z: jnp.ndarray  # [N]
        T: jnp.ndarray  # [N]
        P: jnp.ndarray  # [N]
        SDF: jnp.ndarray  # [N]

        def __add__(self, other):
            return ApproximateSDFLipMLP.DATA(
                X=jnp.concatenate([self.X, other.X], axis=0),
                Y=jnp.concatenate([self.Y, other.Y], axis=0),
                Z=jnp.concatenate([self.Z, other.Z], axis=0),
                T=jnp.concatenate([self.T, other.T], axis=0),
                P=jnp.concatenate([self.P, other.P], axis=0),
                SDF=jnp.concatenate([self.SDF, other.SDF], axis=0),
            )

    def __init__(
        self,
        mlp_layers=(64, 64, 64, 64, 64, 64, 64, 64, 1),
        batch_size=262144,
        model_number=2,
        lip_alpha=0.000001,
        negative_beta=0.0,
    ):
        self.mlp_layers = mlp_layers
        self.batch_size = batch_size
        self.model_number = model_number
        self.negative_beta = negative_beta

        self._init_data = ApproximateSDFLipMLP.DATA(
            X=jnp.zeros(self.batch_size),
            Y=jnp.zeros(self.batch_size),
            Z=jnp.zeros(self.batch_size),
            T=jnp.zeros(self.batch_size),
            P=jnp.zeros((self.batch_size, self.model_number)),
            SDF=jnp.zeros(self.batch_size),
        )
        self.lip_alpha = lip_alpha

    def init_data(self) -> namedtuple:
        return self._init_data

    def init_and_functions(self, rng_key: KeyArray) -> (namedtuple, namedtuple):
        def constructor():
            net = LipMLP(
                output_sizes=self.mlp_layers,
            )

            def f_sdf(x, y, z, t, p):
                vec = jnp.hstack([x, y, z, t, p])
                return jnp.squeeze(net(vec))

            vec_f_sdf = hk.vmap(f_sdf, in_axes=(0, 0, 0, 0, 0), split_rng=False)

            def vec_main_loss(X, Y, Z, T, P, SDF):
                predict = vec_f_sdf(X, Y, Z, T, P)
                return (
                    jnp.mean((predict - SDF) ** 2)
                    + self.negative_beta
                    * jnp.mean((jax.nn.softplus(-predict) - jax.nn.softplus(-SDF)) ** 2)
                    + self.lip_alpha * net.get_lipschitz_loss()
                )

            grad_x = hk.grad(f_sdf, argnums=0)
            grad_y = hk.grad(f_sdf, argnums=1)
            grad_z = hk.grad(f_sdf, argnums=2)
            grad_t = hk.grad(f_sdf, argnums=3)

            vec_grad_x = hk.vmap(grad_x, in_axes=(0, 0, 0, 0, 0), split_rng=False)
            vec_grad_y = hk.vmap(grad_y, in_axes=(0, 0, 0, 0, 0), split_rng=False)
            vec_grad_z = hk.vmap(grad_z, in_axes=(0, 0, 0, 0, 0), split_rng=False)
            vec_grad_t = hk.vmap(grad_t, in_axes=(0, 0, 0, 0, 0), split_rng=False)

            def init(X, Y, Z, T, P, SDF):
                return vec_main_loss(X, Y, Z, T, P, SDF)

            return init, ApproximateSDFLipMLP.FUNC(
                sdf=f_sdf,
                vec_sdf=vec_f_sdf,
                sdf_dx=grad_x,
                sdf_dy=grad_y,
                sdf_dz=grad_z,
                sdf_dt=grad_t,
                vec_sdf_dx=vec_grad_x,
                vec_sdf_dy=vec_grad_y,
                vec_sdf_dz=vec_grad_z,
                vec_sdf_dt=vec_grad_t,
                vec_main_loss=vec_main_loss,
            )

        init, function_tuple = hk.multi_transform(constructor)
        functions = ApproximateSDFLipMLP.FUNC(*function_tuple)
        init_params = init(rng_key, *tuple(self._init_data))

        return init_params, functions


class ApproximateSDFLipMLP2(AbstractTrainableTask):
    """
    This is a trainable task for the problem of shape interpolation between several 3d objects.
    This class employs a multi-layer perceptron with Lipschitz regularization (LipMLP).
    """

    FUNC = namedtuple(
        "ApproximateSDFLipMLP2_DATA",
        [
            "sdf",
            "vec_sdf",
            "sdf_dx",
            "sdf_dy",
            "sdf_dz",
            "sdf_dt",
            "vec_sdf_dx",
            "vec_sdf_dy",
            "vec_sdf_dz",
            "vec_sdf_dt",
            "vec_main_loss",
        ],
    )

    class DATA(NamedTuple):
        X: jnp.ndarray  # [N]
        Y: jnp.ndarray  # [N]
        Z: jnp.ndarray  # [N]
        T: jnp.ndarray  # [N]
        P: jnp.ndarray  # [N]
        SDF: jnp.ndarray  # [N]
        WEIGHT: jnp.ndarray  # [N]

        def __add__(self, other):
            return ApproximateSDFLipMLP2.DATA(
                X=jnp.concatenate([self.X, other.X], axis=0),
                Y=jnp.concatenate([self.Y, other.Y], axis=0),
                Z=jnp.concatenate([self.Z, other.Z], axis=0),
                T=jnp.concatenate([self.T, other.T], axis=0),
                P=jnp.concatenate([self.P, other.P], axis=0),
                SDF=jnp.concatenate([self.SDF, other.SDF], axis=0),
                WEIGHT=jnp.concatenate([self.WEIGHT, other.WEIGHT], axis=0),
            )

    def __init__(
        self,
        mlp_layers=(64, 64, 64, 64, 64, 64, 64, 64, 1),
        batch_size=262144,
        model_number=2,
        lip_alpha=0.000001,
    ):
        self.mlp_layers = mlp_layers
        self.batch_size = batch_size
        self.model_number = model_number

        self._init_data = ApproximateSDFLipMLP2.DATA(
            X=jnp.zeros(self.batch_size),
            Y=jnp.zeros(self.batch_size),
            Z=jnp.zeros(self.batch_size),
            T=jnp.zeros(self.batch_size),
            P=jnp.zeros((self.batch_size, self.model_number)),
            SDF=jnp.zeros(self.batch_size),
            WEIGHT=jnp.ones(self.batch_size),
        )
        self.lip_alpha = lip_alpha

    def init_data(self) -> namedtuple:
        return self._init_data

    def init_and_functions(self, rng_key: KeyArray) -> (namedtuple, namedtuple):
        def constructor():
            net = LipMLP(output_sizes=self.mlp_layers, activation=jnp.sin)

            def f_sdf(x, y, z, t, p):
                vec = jnp.hstack([x, y, z, t, p])
                return jnp.squeeze(net(vec))

            vec_f_sdf = hk.vmap(f_sdf, in_axes=(0, 0, 0, 0, 0), split_rng=False)

            def vec_main_loss(X, Y, Z, T, P, SDF, WEIGHT):
                predict = vec_f_sdf(X, Y, Z, T, P)
                return (
                    jnp.sum(WEIGHT * (predict - SDF) ** 2) / jnp.sum(WEIGHT)
                    + self.lip_alpha * net.get_lipschitz_loss()
                )

            grad_x = hk.grad(f_sdf, argnums=0)
            grad_y = hk.grad(f_sdf, argnums=1)
            grad_z = hk.grad(f_sdf, argnums=2)
            grad_t = hk.grad(f_sdf, argnums=3)

            vec_grad_x = hk.vmap(grad_x, in_axes=(0, 0, 0, 0, 0), split_rng=False)
            vec_grad_y = hk.vmap(grad_y, in_axes=(0, 0, 0, 0, 0), split_rng=False)
            vec_grad_z = hk.vmap(grad_z, in_axes=(0, 0, 0, 0, 0), split_rng=False)
            vec_grad_t = hk.vmap(grad_t, in_axes=(0, 0, 0, 0, 0), split_rng=False)

            def init(X, Y, Z, T, P, SDF, WEIGHT):
                return vec_main_loss(X, Y, Z, T, P, SDF, WEIGHT)

            return init, ApproximateSDFLipMLP2.FUNC(
                sdf=f_sdf,
                vec_sdf=vec_f_sdf,
                sdf_dx=grad_x,
                sdf_dy=grad_y,
                sdf_dz=grad_z,
                sdf_dt=grad_t,
                vec_sdf_dx=vec_grad_x,
                vec_sdf_dy=vec_grad_y,
                vec_sdf_dz=vec_grad_z,
                vec_sdf_dt=vec_grad_t,
                vec_main_loss=vec_main_loss,
            )

        init, function_tuple = hk.multi_transform(constructor)
        functions = ApproximateSDFLipMLP2.FUNC(*function_tuple)
        init_params = init(rng_key, *tuple(self._init_data))

        return init_params, functions


class SurfaceSegmentation(AbstractTrainableTask):
    """
    This is a trainable task for the problem of supervised surface segmentation.
    This class employs the fully-convolutional neural network.
    """

    FUNC = namedtuple("FUNC", ["nn", "main_loss", "metric_accuracy"])
    DATA = namedtuple("DATA", ["SDF_CUBE", "CLASS"])

    def __init__(
        self,
        spacing=(16, 16, 16),
        conv_kernel=32,
        conv_depth=4,
        num_classes=3,
        batch_size=128,
    ):
        self.spacing = spacing
        self.conv_kernel = conv_kernel
        self.conv_depth = conv_depth
        self.num_classes = num_classes
        self.batch_size = batch_size

        self._init_data = SurfaceSegmentation.DATA(
            SDF_CUBE=jnp.zeros(
                (self.batch_size, spacing[0], spacing[1], spacing[2], 1)
            ),
            CLASS=jnp.zeros(self.batch_size),
        )

    def init_data(self) -> namedtuple:
        return self._init_data

    def init_and_functions(self, rng_key: KeyArray) -> (namedtuple, namedtuple):
        def constructor():
            descendants_convolution = DescConv(
                n_layers=self.conv_depth,
                kernels_in_first_layer=self.conv_kernel,
                kernel_shape=(2, 2, 2),
                stride=(2, 2, 2),
                activation=jax.nn.relu,
            )

            def nn(input_):  # NDHWC
                x = input_
                x = descendants_convolution(x)

                x = hk.Conv3D(
                    output_channels=self.conv_kernel * 4,
                    kernel_shape=(1, 1, 1),
                    padding="VALID",
                    name="mlp_0",
                )(x)
                x = jax.nn.relu(x)
                x = hk.Conv3D(
                    output_channels=self.conv_kernel * 4,
                    kernel_shape=(1, 1, 1),
                    padding="VALID",
                    name="mlp_1",
                )(x)
                x = jax.nn.relu(x)

                x = hk.Conv3D(
                    output_channels=self.num_classes,
                    kernel_shape=(1, 1, 1),
                    padding="VALID",
                    name="mlp_output_0",
                )(x)
                x = jax.nn.softmax(x)

                return jnp.squeeze(x)

            def softmax_cross_entropy(logits, labels):
                one_hot = jax.nn.one_hot(labels, self.num_classes)
                return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)

            def metric_accuracy(sdf_tiles, labels):
                logits = nn(sdf_tiles)
                return jnp.mean(jnp.argmax(logits, -1) == labels)

            def main_loss(sdf_tiles, labels):
                logits = nn(sdf_tiles)
                return jnp.mean(softmax_cross_entropy(logits, labels))

            def init(sdf_tiles, labels):
                return main_loss(sdf_tiles, labels)

            return init, SurfaceSegmentation.FUNC(
                nn=nn, main_loss=main_loss, metric_accuracy=metric_accuracy
            )

        init, function_tuple = hk.multi_transform(constructor)
        functions = SurfaceSegmentation.FUNC(*function_tuple)
        init_params = init(rng_key, *tuple(self._init_data))

        return init_params, functions


class Eikonal3D(AbstractTrainableTask):
    """
    This is a solver for the Eikonal equation.
    """

    class FUNC(NamedTuple):
        nn: Callable[[float, float, float], float]  # [N]
        nn_dx: Callable[[float, float, float], float]  # [N]
        nn_dy: Callable[[float, float, float], float]  # [N]
        nn_dz: Callable[[float, float, float], float]  # [N]
        main_loss: Callable[[float, float, float], float]  # [N]

    class DATA(NamedTuple):  # F(X,Y,Z) = U
        X: jnp.ndarray  # [N]
        Y: jnp.ndarray  # [N]
        Z: jnp.ndarray  # [N]

    def __init__(
        self,
        fun_sdf_domain: Callable[[float, float, float], float],
        fun_sdf_start: Callable[[float, float, float], float],
        mlp_layers=(
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            1,
        ),
        lambda_grad=0.1,
        lambda_non_negativity=0.1,
        batch_size=100,
    ):
        self.fun_sdf_domain = fun_sdf_domain
        self.fun_sdf_start = fun_sdf_start

        self.mlp_layers = mlp_layers
        self.batch_size = batch_size

        self._init_data = Eikonal3D.DATA(
            X=jnp.zeros(self.batch_size),
            Y=jnp.zeros(self.batch_size),
            Z=jnp.zeros(self.batch_size),
        )

        self.LAMBDA_GRAD = lambda_grad
        self.LAMBDA_NON_NEGATIVITY = lambda_non_negativity

    def init_data(self) -> namedtuple:
        return self._init_data

    def init_and_functions(self, rng_key: KeyArray) -> (namedtuple, namedtuple):
        mlp_layers = self.mlp_layers
        fun_sdf_domain = self.fun_sdf_domain
        fun_sdf_start = self.fun_sdf_start

        LAMBDA_NON_NEGATIVITY = self.LAMBDA_NON_NEGATIVITY
        LAMBDA_GRAD = self.LAMBDA_GRAD

        def constructor():
            def NN(x, y, z):
                vec = jnp.hstack([x, y, z])
                net = hk.nets.MLP(output_sizes=mlp_layers, activation=jnp.tanh)
                return jnp.squeeze(net(vec))

            vec_NN = hk.vmap(NN, in_axes=(0, 0, 0), split_rng=False)

            grad_x = hk.grad(NN, argnums=0)
            grad_y = hk.grad(NN, argnums=1)
            grad_z = hk.grad(NN, argnums=2)

            vec_grad_x = hk.vmap(grad_x, in_axes=(0, 0, 0), split_rng=False)
            vec_grad_y = hk.vmap(grad_y, in_axes=(0, 0, 0), split_rng=False)
            vec_grad_z = hk.vmap(grad_z, in_axes=(0, 0, 0), split_rng=False)

            def vec_conductivity(X, Y, Z):
                return 1.0 * (fun_sdf_domain(X, Y, Z) < 0)

            def vec_eikonal(x, y, z):
                return (
                    vec_grad_x(x, y, z) ** 2
                    + vec_grad_y(x, y, z) ** 2
                    + vec_grad_z(x, y, z) ** 2
                )

            def vec_region0(X, Y, Z, SDF):
                return 1.0 * (fun_sdf_start(X, Y, Z) < 0)

            def vec_main_loss(X, Y, Z):
                Omega0 = fun_sdf_start(X, Y, Z) < 0
                Omega1 = 1.0 - Omega0

                pred_u = vec_NN(X, Y, Z)

                loss = (
                    jnp.mean(Omega0 * (0.0 - pred_u) ** 2)
                    + LAMBDA_GRAD
                    * jnp.mean(
                        Omega1 * (vec_eikonal(X, Y, Z) - vec_conductivity(X, Y, Z)) ** 2
                    )
                    + LAMBDA_NON_NEGATIVITY * jnp.mean(jax.nn.relu(-pred_u) ** 2)
                )

                return loss

            def init(X, Y, Z):
                return vec_main_loss(X, Y, Z)

            return init, Eikonal3D.FUNC(
                nn=vec_NN,
                nn_dx=vec_grad_x,
                nn_dy=vec_grad_y,
                nn_dz=vec_grad_z,
                main_loss=vec_main_loss,
            )

        init, function_tuple = hk.multi_transform(constructor)
        functions = Eikonal3D.FUNC(*function_tuple)
        init_params = init(rng_key, *tuple(self._init_data))

        return init_params, functions
