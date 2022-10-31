import fnmatch
import os
import warnings
from typing import Dict, Optional, Sequence, Union

import jax
import jax.numpy as jnp
from anytree.exporter import DictExporter, JsonExporter
from anytree.importer import DictImporter, JsonImporter
from jax.random import KeyArray

from nndt.math_core import train_test_split
from nndt.primitive_sdf import SphereSDF
from nndt.space2 import AbstractBBoxNode, AbstractTreeElement
from nndt.space2.filesource import FileSource
from nndt.space2.group import Group
from nndt.space2.implicit_representation import ImpRepr
from nndt.space2.method_set import SamplingMethodSetNode, SDTMethodSetNode
from nndt.space2.object3D import Object3D
from nndt.space2.space import Space
from nndt.space2.transformation import IdentityTransform
from nndt.space2.tree_utils import update_bbox_with_float_over_tree


def _attribute_filter(attrs):
    ret = [(k, v) for k, v in attrs if isinstance(v, (int, float, str, tuple))]
    return sorted(ret)


def _children_filter(children):
    ret = [v for v in children if isinstance(v, AbstractBBoxNode)]
    return ret


def _nodecls_function(parent=None, **attrs):
    if "_nodetype" not in attrs:
        raise ValueError("_nodetype is not located in some node of space file")

    if attrs["_nodetype"] == "FS":
        ret = DICT_NODETYPE_CLASS[attrs["_nodetype"]](
            attrs["name"], attrs["filepath"], attrs["loader_type"], parent=parent
        )
    else:
        ret = DICT_NODETYPE_CLASS[attrs["_nodetype"]](attrs["name"], parent=parent)

    return ret


def load_txt(fullpath):
    """Load Space from txt file.

    Args:
        fullpath (str): Path to txt file.

    Returns:
        Space: initialized space.
    """
    return load_only_one_file(fullpath, loader_type="txt")


def load_sdt(fullpath):
    """Load Space from signed distance tensor file.

    Args:
        fullpath (str): Path to SDT file.

    Returns:
        Space: initialized space.
    """
    return load_only_one_file(fullpath, loader_type="sdt")


def load_mesh_obj(fullpath):
    """Load Space from mesh object file.

    Args:
        fullpath (str): Path to SDT file.

    Returns:
        Space: initialized space.
    """
    return load_only_one_file(fullpath, loader_type="mesh_obj")


def load_implicit_ir1(fullpath):
    return load_only_one_file(fullpath, loader_type="implicit_ir1")


def load_only_one_file(fullpath, loader_type="txt"):
    """Load one file. Allowed types of file: txt, sdt, mesh_obj.

    Args:
        fullpath (str): Path to file.
        loader_type (str, optional): Loader type. Allowed: txt, sdt, mes_obj. Defaults to "txt".

    Raises:
        ValueError: loader_type is unknown.

    Returns:
        Space: initialized space.
    """
    if loader_type not in ["txt", "sdt", "mesh_obj"]:
        raise ValueError("loader_type is unknown")

    space = Space("space")
    default = Object3D("default", parent=space)
    _filesource = FileSource(
        os.path.basename(fullpath), fullpath, loader_type, parent=default
    )
    space.init()
    return space


def load_from_path(
    root_path,
    template_txt="*.txt",
    template_sdt="*sd[ft]*.npy",
    template_mesh_obj="*.obj",
    template_implicit_ir1="*.ir1",
):
    if not os.path.exists(root_path):
        raise FileNotFoundError(
            f"Path {root_path} is not exist. Check relative path or folder presence."
        )

    loader_type_array = []
    space = Space("space")

    def filename_to_loader_type(filename):
        if (template_txt is not None) and fnmatch.fnmatch(filename, template_txt):
            ret = "txt"
        elif (template_sdt is not None) and fnmatch.fnmatch(filename, template_sdt):
            ret = "sdt"
        elif (template_mesh_obj is not None) and fnmatch.fnmatch(
            filename, template_mesh_obj
        ):
            ret = "mesh_obj"
        elif (template_implicit_ir1 is not None) and fnmatch.fnmatch(
            filename, template_implicit_ir1
        ):
            ret = "implicit_ir1"
        else:
            warnings.warn("Some file in path is ignored")
            ret = None
        return ret

    def add_values(lst, fullpath, filename):
        if len(lst) >= 2:
            current_node_ = space
            for ind, name_ in enumerate(lst):
                child_names = [child.name for child in current_node_.children]
                if name_ in child_names:
                    current_node_ = current_node_[name_]
                else:
                    if ind < (len(lst) - 2):
                        current_node_ = Group(name_, parent=current_node_)
                    elif ind == (len(lst) - 2):
                        current_node_ = Object3D(name_, parent=current_node_)
                    elif ind == (len(lst) - 1):
                        loader_type = filename_to_loader_type(filename)
                        if loader_type is not None:
                            loader_type_array.append(loader_type)
                            current_node_ = FileSource(
                                name_, fullpath, loader_type, parent=current_node_
                            )

    for root, dirs, files in os.walk(root_path, topdown=False):
        for fl in files:
            line = os.path.relpath(root, root_path)
            line2 = os.path.join(line, fl)
            lst = line2.split("/")
            add_values(lst, os.path.join(root, fl), fl)

    if ("sdt" in loader_type_array) and ("implicit_ir1" in loader_type_array):
        raise NotImplementedError("SDT and SDF simultaneously are not supported yet!")

    space.init()
    return space


def load_from_file_lists(
    name_list,
    mesh_list: Optional[Sequence[str]] = None,
    sdt_list: Optional[Sequence[str]] = None,
    ir1_list: Optional[Sequence[str]] = None,
    test_size: Optional[float] = None,
) -> Space:
    if mesh_list is not None:
        assert len(name_list) == len(mesh_list)
    if sdt_list is not None:
        assert len(name_list) == len(sdt_list)
    if ir1_list is not None:
        assert len(name_list) == len(ir1_list)

    if (sdt_list is not None) and (ir1_list is not None):
        raise NotImplementedError("SDT and SDF simultaneously are not supported yet!")

    if test_size is None:
        space = Space("main")
        group = Group("default", parent=space)
        for ind, name in enumerate(name_list):
            object_ = Object3D(name, parent=group)
            if mesh_list is not None:
                mesh_source = FileSource(
                    os.path.basename(mesh_list[ind]),
                    mesh_list[ind],
                    "mesh_obj",
                    parent=object_,
                )
            if sdt_list is not None:
                sdt_source = FileSource(
                    os.path.basename(sdt_list[ind]),
                    sdt_list[ind],
                    "sdt",
                    parent=object_,
                )
    else:
        assert 0.0 < test_size < 1.0
        space = Space("main")
        rng_key = jax.random.PRNGKey(42)
        index_train, index_test = train_test_split(
            range(len(name_list)), rng_key, test_size=test_size
        )

        group_train = Group("train", parent=space)
        for ind in index_train:
            name = name_list[ind]
            object_ = Object3D(name, parent=group_train)
            if mesh_list is not None:
                mesh_source = FileSource(
                    os.path.basename(mesh_list[ind]),
                    mesh_list[ind],
                    "mesh_obj",
                    parent=object_,
                )
            if sdt_list is not None:
                sdt_source = FileSource(
                    os.path.basename(sdt_list[ind]),
                    sdt_list[ind],
                    "sdt",
                    parent=object_,
                )
            if ir1_list is not None:
                ir1_source = FileSource(
                    os.path.basename(ir1_list[ind]),
                    ir1_list[ind],
                    "implicit_ir1",
                    parent=object_,
                )

        group_test = Group("test", parent=space)
        for ind in index_test:
            name = name_list[ind]
            object_ = Object3D(name, parent=group_test)
            if mesh_list is not None:
                mesh_source = FileSource(
                    os.path.basename(mesh_list[ind]),
                    mesh_list[ind],
                    "mesh_obj",
                    parent=object_,
                )
            if sdt_list is not None:
                sdt_source = FileSource(
                    os.path.basename(sdt_list[ind]),
                    sdt_list[ind],
                    "sdt",
                    parent=object_,
                )
            if ir1_list is not None:
                ir1_source = FileSource(
                    os.path.basename(ir1_list[ind]),
                    ir1_list[ind],
                    "implicit_ir1",
                    parent=object_,
                )

    space.init()
    return space


def read_space_from_file(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError()

    dict_imp = DictImporter(nodecls=_nodecls_function)
    json_imp = JsonImporter(dictimporter=dict_imp)
    with open(filepath, "r") as fl:
        space = json_imp.read(fl)

    space.init()
    return space


def from_json(json: str):
    dict_imp = DictImporter(nodecls=_nodecls_function)
    json_imp = JsonImporter(dictimporter=dict_imp)
    space = json_imp.import_(json)

    space.init()
    return space


def save_space_to_file(space: Space, filepath: str):
    filepath_with_ext = filepath if filepath.endswith(".space") else filepath + ".space"
    with open(filepath_with_ext, "w", encoding="utf-8") as fl:
        fl.write(space.to_json())


def to_json(space: Space):
    dict_exp = DictExporter(attriter=_attribute_filter, childiter=_children_filter)
    json_exp = JsonExporter(dictexporter=dict_exp, indent=2)
    return json_exp.export(space)


def __reconstruct_tree_one_node(
    tree_path: AbstractBBoxNode, nodelist: list, nodename: str
):
    from nndt.space2.space_preloader import _update_bbox_bottom_to_up

    train_node = Group(nodename, parent=tree_path)
    for node in nodelist:
        node.parent = train_node
    _update_bbox_bottom_to_up(train_node)


def __reconstruct_tree(tree_path: AbstractBBoxNode, train: list, test: list):
    __reconstruct_tree_one_node(tree_path, train, "train")
    __reconstruct_tree_one_node(tree_path, test, "test")
    tree_path.root.init()
    return tree_path.root


def split_node_test_train(
    rng_key: KeyArray, tree_path: AbstractBBoxNode, test_size: float = 0.3
):
    """
    Split node to two nodes according to requested proportion. This method creates nodes with test and train names.
    New nodes are attached as new groups.

    :param rng_key: a key for JAX's random generators
    :param tree_path: node of the space model tree for the split
    :param test_size: size of the test subset. value must range between 0 and 1.
    :return: the root of the space model tree
    """
    child_ = tree_path._container_only_list()
    indices = jnp.arange(len(child_))

    train_index_list, test_index_list = train_test_split(
        indices, rng=rng_key, test_size=test_size
    )
    train = [child_[i] for i in train_index_list]
    test = [child_[i] for i in test_index_list]

    root = __reconstruct_tree(tree_path, train, test)
    return root


def split_node_kfold(
    tree_path: AbstractBBoxNode,
    n_fold: int = 5,
    k_for_test: Union[Sequence[int], int] = 0,
):
    """
    Split node to several nodes according to k-fold approach. This method creates nodes with test and train names.
    New nodes are attached as new groups.

    :param tree_path: node of the space model tree for the split
    :param n_fold: number of folds
    :param k_for_test: index of list of indexes that were attached as the test group
    :return: the root of the space model tree
    """
    child_ = tree_path._container_only_list()
    assert len(child_) >= n_fold
    folds = jnp.array_split(jnp.arange(len(child_), dtype=int), n_fold)
    k_lst = [k_for_test] if isinstance(k_for_test, int) else list(k_for_test)
    train_ind = []
    test_ind = []

    for fold_i, fold_tpl in enumerate(folds):
        for i in fold_tpl:
            if fold_i in k_lst:
                test_ind.append(i)
            else:
                train_ind.append(i)

    train = [child_[i] for i in train_ind]
    test = [child_[i] for i in test_ind]

    root = __reconstruct_tree(tree_path, train, test)
    return root


def split_node_namelist(
    tree_path: AbstractBBoxNode, dict_nodename_namelist: Dict[str, Sequence[str]]
):
    """
    Split node to several nodes according to the dictionary. New nodes are attached as new groups.

    :param tree_path: node of the space model tree for the split
    :param dict_nodename_namelist: key is a name for the new group. value is a list of children for the new group.
    :return: the root of the space model tree
    """
    lst_nodenames = [child.name for child in tree_path]
    lst_temp = [child.name for child in tree_path]
    for nodename, namelist in dict_nodename_namelist.items():
        for name in namelist:
            if name not in lst_nodenames:
                raise ValueError(
                    f"Name {name} is not a valid child of {tree_path.name}"
                )
            if name not in lst_temp:
                raise ValueError(f"Name {name} is duplicated in dict_nodename_namelist")
            else:
                lst_temp.remove(name)

    if len(lst_temp):
        raise ValueError(
            f"The following names are not mentioned in dict_nodename_namelist: {lst_temp}"
        )

    for nodename, namelist in dict_nodename_namelist.items():
        __reconstruct_tree_one_node(
            tree_path, [tree_path[name] for name in namelist], nodename
        )

    tree_path.root.init()
    return tree_path.root


def add_sphere(
    tree_path: AbstractBBoxNode, name, center: (float, float, float), radius: float
):
    assert isinstance(tree_path, (Space, Group))

    sph = SphereSDF(center=center, radius=radius)

    obj = Object3D(name, bbox=sph.bbox, parent=tree_path)
    transform = IdentityTransform(ps_bbox=sph.bbox, parent=obj)
    ir = ImpRepr("sphere", sph, parent=obj)
    ms = SDTMethodSetNode(obj, ir, transform, parent=obj)
    smp = SamplingMethodSetNode(parent=obj)

    update_bbox_with_float_over_tree(obj)
    tree_path.root.init()
    return tree_path.root


DICT_NODETYPE_CLASS = {
    "UNDEFINED": AbstractTreeElement,
    "S": Space,
    "G": Group,
    "O3D": Object3D,
    "FS": FileSource,
}
DICT_CLASS_NODETYPE = {(v, k) for k, v in DICT_NODETYPE_CLASS.items()}
