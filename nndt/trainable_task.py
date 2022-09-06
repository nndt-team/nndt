from abc import abstractmethod
from collections import namedtuple
from typing import NamedTuple

import haiku as hk
import jax.numpy as jnp
from jax.random import KeyArray


class AbstractTrainableTask:

    @abstractmethod
    def init_data(self) -> namedtuple:
        pass

    @abstractmethod
    def init_and_functions(self, rng_key: KeyArray) -> (namedtuple, namedtuple):
        pass


class ApproximateSDF(AbstractTrainableTask):
    FUNC = namedtuple("ApproximateSDF_DATA", ["sdf", "vec_sdf",
                                              "sdf_dx", "sdf_dy", "sdf_dz", "sdf_dt",
                                              "vec_sdf_dx", "vec_sdf_dy", "vec_sdf_dz", "vec_sdf_dt",
                                              "vec_main_loss"])

    # DATA = namedtuple("ApproximateSDF_FUNC", ["X", "Y", "Z", "T", "P", "SDF"])

    class DATA(NamedTuple):
        X: jnp.ndarray  # [N]
        Y: jnp.ndarray  # [N]
        Z: jnp.ndarray  # [N]
        T: jnp.ndarray  # [N]
        P: jnp.ndarray  # [N]
        SDF: jnp.ndarray  # [N]

        def __add__(self, other):
            return ApproximateSDF.DATA(X=jnp.concatenate([self.X, other.X], axis=0),
                                       Y=jnp.concatenate([self.Y, other.Y], axis=0),
                                       Z=jnp.concatenate([self.Z, other.Z], axis=0),
                                       T=jnp.concatenate([self.T, other.T], axis=0),
                                       P=jnp.concatenate([self.P, other.P], axis=0),
                                       SDF=jnp.concatenate([self.SDF, other.SDF], axis=0))

    def __init__(self,
                 mlp_layers=(64, 64, 64, 64, 64, 64, 64, 64, 1),
                 batch_size=262144,
                 model_number=2):
        self.mlp_layers = mlp_layers
        self.batch_size = batch_size
        self.model_number = model_number

        self._init_data = ApproximateSDF.DATA(X=jnp.zeros(self.batch_size),
                                              Y=jnp.zeros(self.batch_size),
                                              Z=jnp.zeros(self.batch_size),
                                              T=jnp.zeros(self.batch_size),
                                              P=jnp.zeros((self.batch_size, self.model_number)),
                                              SDF=jnp.zeros(self.batch_size))

    def init_data(self) -> namedtuple:
        return self._init_data

    def init_and_functions(self, rng_key: KeyArray) -> (namedtuple, namedtuple):
        def constructor():
            def f_sdf(x, y, z, t, p):
                vec = jnp.hstack([x, y, z, t, p])
                net = hk.nets.MLP(output_sizes=self.mlp_layers,
                                  activation=jnp.tanh)
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

            return init, ApproximateSDF.FUNC(sdf=f_sdf, vec_sdf=vec_f_sdf,
                                             sdf_dx=grad_x, sdf_dy=grad_y, sdf_dz=grad_z, sdf_dt=grad_t,
                                             vec_sdf_dx=vec_grad_x, vec_sdf_dy=vec_grad_y, vec_sdf_dz=vec_grad_z,
                                             vec_sdf_dt=vec_grad_t,
                                             vec_main_loss=vec_main_loss)

        init, function_tuple = hk.multi_transform(constructor)
        functions = ApproximateSDF.FUNC(*function_tuple)
        init_params = init(rng_key, *tuple(self._init_data))

        return init_params, functions
