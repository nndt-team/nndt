import haiku as hk
import jax
import jax.numpy as jnp
from typing import *


class DescConv(hk.Module):

    def __init__(self,
                 n_layers=4, kernels_in_first_layer=32,
                 kernel_shape=(2, 2, 2),
                 stride=(2, 2, 2),
                 activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.n_layers = n_layers
        self.kernels_in_first_layer = kernels_in_first_layer
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.activation = activation

        layers = []
        for index in range(n_layers):
            layers.append(hk.Conv3D(
                            output_channels=2**index * kernels_in_first_layer,
                            kernel_shape=kernel_shape,
                            stride=stride,
                            w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
                            b_init=jnp.zeros,
                            padding="VALID"))

        self.layers = tuple(layers)

    def __call__(self, inputs):
        out = inputs
        activation = self.activation
        for i, layer in enumerate(self.layers):
            out = layer(out)
            out = activation(out)
        return out
