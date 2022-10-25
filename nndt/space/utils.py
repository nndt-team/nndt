from typing import Union

from nndt.space.abstracts import AbstractRegion, ExtendedNodeMixin


def downup_update_bbox(leaf: Union[AbstractRegion, ExtendedNodeMixin]):
    (Xmin, Ymin, Zmin), (Xmax, Ymax, Zmax) = leaf._bbox

    current = leaf
    while True:
        if hasattr(current, "_bbox"):
            (Xmin2, Ymin2, Zmin2), (Xmax2, Ymax2, Zmax2) = current._bbox
            current._bbox = (
                (min(Xmin, Xmin2), min(Ymin, Ymin2), min(Ymin, Ymin2)),
                (max(Xmax, Xmax2), max(Ymax, Ymax2), max(Ymax, Ymax2)),
            )
            # TODO THIS IS WRONG!
        if current.is_root:
            break
        current = current.parent
