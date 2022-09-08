from typing import Union
import jax
import jax.numpy as jnp
import jax.random as random

from space.abstracts import AbstractRegion, ExtendedNodeMixin


def downup_update_bbox(leaf: Union[AbstractRegion, ExtendedNodeMixin]):
    (Xmin, Ymin, Zmin), (Xmax, Ymax, Zmax) = leaf._bbox

    current = leaf
    while True:
        if hasattr(current, '_bbox'):
            (Xmin2, Ymin2, Zmin2), (Xmax2, Ymax2, Zmax2) = current._bbox
            current._bbox = ((min(Xmin, Xmin2), min(Ymin, Ymin2), min(Ymin, Ymin2)),
                             (max(Xmax, Xmax2), max(Ymax, Ymax2), max(Ymax, Ymax2)))
        if current.is_root:
            break
        current = current.parent


def train_test_split(array,
                     rng: jax.random.PRNGKey,
                     test_size: float = 0.3) -> (list, list):

    _, rng = random.split(rng)
    indices = jnp.arange(len(array))
    test_ind = jax.random.choice(key=rng, a=indices, replace=False, shape=[int(len(indices)*test_size)]).tolist()

    test_index_list = [index for index in test_ind]
    train_index_list = jnp.array([item for item in indices if item not in test_index_list])\
        .reshape(int(jnp.ceil(len(indices) * (1 - test_size))))\
        .tolist()

    return train_index_list, test_index_list
