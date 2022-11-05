from anytree import NodeMixin

from nndt.space2.abstracts import AbstractBBoxNode
from nndt.space2.utils import update_bbox


def update_bbox_from_children(node):
    """
    Update the boundary box size for the node according to the boundary box size of all children.

    Args:
        node : Node to update.
    """

    for child in node.children:
        if isinstance(child, AbstractBBoxNode):
            node.bbox = update_bbox(node.bbox, child.bbox)


def update_bbox_with_float_over_tree(node: NodeMixin):
    """
    Update box size for node.
    This method iterates over all parents and ancestors of the node and updates the boundary box according to the node boundary box.

    Args:
        node (NodeMixin): start node.
    """

    current_node = node
    while not current_node.is_root:
        bbox1 = current_node.bbox
        current_node = current_node.parent
        current_node.bbox = update_bbox(current_node.bbox, bbox1)
