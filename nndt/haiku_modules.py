from typing import *

import haiku as hk
import jax
import jax.numpy as jnp


class DescConv(hk.Module):
    """
    Fully convolutional network
    """

    def __init__(
        self,
        n_layers=4,
        kernels_in_first_layer=32,
        kernel_shape=(2, 2, 2),
        stride=(2, 2, 2),
        activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.n_layers = n_layers
        self.kernels_in_first_layer = kernels_in_first_layer
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.activation = activation

        layers = []
        for index in range(n_layers):
            layers.append(
                hk.Conv3D(
                    output_channels=2**index * kernels_in_first_layer,
                    kernel_shape=kernel_shape,
                    stride=stride,
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
                    b_init=jnp.zeros,
                    padding="VALID",
                )
            )

        self.layers = tuple(layers)

    def __call__(self, inputs):
        out = inputs
        activation = self.activation
        for i, layer in enumerate(self.layers):
            out = layer(out)
            out = activation(out)
        return out


class LipLinear(hk.Module):
    """
    Layer for implementation of the LipMLP from the article:
    Liu, Hsueh-Ti Derek, et al. "Learning Smooth Neural Functions via Lipschitz Regularization."
    arXiv preprint arXiv:2202.08345 (2022).
    """

    def __init__(
        self,
        output_size,
        name=None,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
    ):
        super().__init__(name=name)
        self.output_size = output_size
        self.activation = activation

    def weight_normalization(self, W, softplus_c):
        absrowsum = jnp.sum(jnp.abs(W), axis=1)
        scale = jnp.minimum(1.0, softplus_c / absrowsum)
        return W * scale[:, None]

    def __call__(self, x):
        input_size, output_size = x.shape[-1], self.output_size
        w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "truncated_normal")
        W = hk.get_parameter(
            "W", shape=[input_size, output_size], dtype=x.dtype, init=w_init
        )
        b = hk.get_parameter("b", shape=[output_size], dtype=x.dtype, init=jnp.zeros)

        def _c_init(shape: Sequence[int], dtype: Any) -> jnp.ndarray:
            return jnp.max(jnp.sum(jnp.abs(W.T), axis=1))

        c = hk.get_parameter("c", shape=(), dtype=x.dtype, init=_c_init)
        W_ = self.weight_normalization(W.T, jax.nn.softplus(c)).T
        out = self.activation(jnp.dot(x, W_) + b)
        return out

    def get_lipschitz_loss(self):
        c = hk.get_parameter("c", shape=(), init=jnp.zeros)
        return jax.nn.softplus(c)


class LipMLP(hk.Module):
    """
    This is an implementation of the LipMLP from the article:
    Liu, Hsueh-Ti Derek, et al. "Learning Smooth Neural Functions via Lipschitz Regularization."
    arXiv preprint arXiv:2202.08345 (2022).
    """

    def __init__(
        self,
        output_sizes: Iterable[int],
        name: Optional[str] = None,
        activation=jax.nn.tanh,
        activation_output=lambda x: x,
    ):
        super().__init__(name=name)
        self.output_sizes = output_sizes

        layers = []
        output_sizes = tuple(output_sizes)
        for index, output_size in enumerate(output_sizes[:-1]):
            layers.append(
                LipLinear(
                    output_size=output_size,
                    name="lip_mlp_%d" % index,
                    activation=activation,
                )
            )
        index, output_size = len(output_sizes) - 1, output_sizes[-1]
        layers.append(
            LipLinear(
                output_size=output_size,
                name="lip_mlp_%d" % index,
                activation=activation_output,
            )
        )

        self.layers = tuple(layers)

    def __call__(self, inputs):
        out = inputs
        for i, layer in enumerate(self.layers):
            out = layer(out)
        return out

    def get_lipschitz_loss(self):
        out = 1.0
        for i, layer in enumerate(self.layers):
            out = out * layer.get_lipschitz_loss()
        return out
