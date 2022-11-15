from typing import Optional

from anytree import RenderTree

from nndt.space2.abstracts import AbstractBBoxNode, AbstractTreeElement
from nndt.space2.filesource import FileSource
from nndt.space2.implicit_representation import ImpRepr
from nndt.space2.method_set import MethodSetNode
from nndt.space2.transformation import AbstractTransformation


def _construct_filter(child_classes, not_parent_classes):
    def filter_(children):
        ret = [
            v
            for v in children
            if isinstance(v, child_classes)
            and not isinstance(v.parent, not_parent_classes)
        ]
        return ret

    return filter_


class PrintContainer:
    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return self.text

    def _repr_html_(self):
        return f'<p>{self.text}</p>'

    def __str__(self):
        return self.text


def _pretty_print(node: AbstractTreeElement, mode: Optional[str] = "default"):
    if mode is None or (mode == "default"):
        ret = RenderTree(
            node,
            childiter=_construct_filter(
                (AbstractTreeElement,),
                (MethodSetNode, AbstractTransformation, FileSource, ImpRepr),
            ),
        ).__str__()
    elif mode == "source" or mode == "sources":
        ret = RenderTree(
            node, childiter=_construct_filter((AbstractBBoxNode,), ())
        ).__str__()
    elif mode == "full":
        ret = RenderTree(node).__str__()
    else:
        raise NotImplementedError(f"{mode} is not implemented for the explore method. ")
    return PrintContainer(ret)
