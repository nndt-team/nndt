from typing import Optional

from anytree import RenderTree

from nndt.space2 import AbstractTreeElement, AbstractBBoxNode, MethodSetNode, AbstractTransformation, FileSource


def _construct_filter(child_classes, not_parent_classes):
    def filter_(children):
        ret = [v for v in children
               if isinstance(v, child_classes) and
                not isinstance(v.parent, not_parent_classes)]
        return ret

    return filter_

def _pretty_print(node: AbstractTreeElement, mode: Optional[str] = "default"):
    if mode is None or (mode == "default"):
        ret = RenderTree(node,
                         childiter=_construct_filter((AbstractTreeElement,),
                                                     (MethodSetNode, AbstractTransformation, FileSource))).__str__()
    elif mode == "source" or mode == "sources":
        ret = RenderTree(node,
                         childiter=_construct_filter((AbstractBBoxNode,),
                                                     ())).__str__()
    elif mode == "full":
        ret = RenderTree(node).__str__()
    else:
        raise NotImplementedError(f"{mode} is not implemented for the explore method. ")
    return ret