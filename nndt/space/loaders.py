import gc
from anytree import PreOrderIter
from sklearn.model_selection import train_test_split

from nndt.space.regions import *
from nndt.space.repr_mesh import *
from nndt.space.repr_prim import SphereSDF_Xyz2SDT, SphereSDF, SphereSDF_PureSDF
from nndt.space.repr_sdt import *
from nndt.space.sources import SphereSDFSource


def preload_all_possible(space: Space,
                         padding_physical=(10, 10, 10),
                         scale_physical2normed=50):
    for node in PreOrderIter(space):
        if isinstance(node, AbstractRegion):
            _ = SamplingGrid(node)
            _ = SamplingUniform(node)
            _ = SamplingGridWithShackle(node)

        if isinstance(node, MeshSource):
            mesh_repr = MeshRepr.load_mesh_and_bring_to_center(node,
                                                               padding_physical=padding_physical,
                                                               scale_physical2normed=scale_physical2normed)
            _ = Index2xyz(mesh_repr)
            _ = SamplingEachN(mesh_repr)
            _ = SaveMesh(mesh_repr)

            color_reps = PointColorRepr.try_to_build_representation(mesh_repr)

        if isinstance(node, SDTSource):
            repr = SDTRepr.load_mesh_and_bring_to_center(node,
                                                         padding_physical=padding_physical,
                                                         scale_physical2normed=scale_physical2normed)

            _ = Xyz2SDT(repr)
            _ = Xyz2LocalSDT(repr)

        if isinstance(node, SphereSDFSource):
            repr = SphereSDF(node, name="repr")
            _ = SphereSDF_Xyz2SDT(repr)
            _ = SphereSDF_PureSDF(repr)

    gc.collect()


def load_data(name_list,
              mesh_list: Optional[Sequence[str]] = None,
              sdt_list: Optional[Sequence[str]] = None,
              test_size: Optional[float] = None) -> Space:
    if mesh_list is not None:
        assert (len(name_list) == len(mesh_list))
    if sdt_list is not None:
        assert (len(name_list) == len(sdt_list))

    if test_size is None:
        space = Space("main")
        group = Group("default", parent=space)
        for ind, name in enumerate(name_list):
            object = Object(name, parent=group)
            if mesh_list is not None:
                mesh_source = MeshSource("mesh", mesh_list[ind], parent=object)
            if sdt_list is not None:
                sdt_source = SDTSource("sdt", sdt_list[ind], parent=object)
    else:
        assert (0.0 < test_size < 1.0)
        space = Space("main")
        index_train, index_test = train_test_split(range(len(name_list)),
                                                   test_size=test_size,
                                                   random_state=42)

        group_train = Group("train", parent=space)
        for ind in index_train:
            name = name_list[ind]
            object = Object(name, parent=group_train)
            if mesh_list is not None:
                mesh_source = MeshSource("mesh", mesh_list[ind], parent=object)
            if sdt_list is not None:
                sdt_source = SDTSource("sdt", sdt_list[ind], parent=object)

        group_test = Group("test", parent=space)
        for ind in index_test:
            name = name_list[ind]
            object = Object(name, parent=group_test)
            if mesh_list is not None:
                mesh_source = MeshSource("mesh", mesh_list[ind], parent=object)
            if sdt_list is not None:
                sdt_source = SDTSource("sdt", sdt_list[ind], parent=object)

    return space
