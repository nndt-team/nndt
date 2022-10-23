from anytree import PostOrderIter, PreOrderIter

from nndt.space2 import ColorMethodSetNode
from nndt.space2 import SDTMethodSetNode
from nndt.space2 import SamplingMethodSetNode
from nndt.space2 import pad_bbox
from nndt.space2 import update_bbox, AbstractBBoxNode, Space, FileSource, Object3D, Group, AbstractTransformation, \
    MeshObjMethodSetNode, \
    DICT_NODETYPE_PRIORITY
from nndt.space2.method_set_train_task import TrainTaskSetNode


def _update_bbox_bottom_to_up(node):
    for child in node.children:
        if isinstance(child, AbstractBBoxNode):
            node.bbox = update_bbox(node.bbox, child.bbox)


class DefaultPreloader:

    def __init__(self,
                 mode="identity",
                 scale=50,
                 keep_in_memory=True,
                 ps_padding=(0., 0., 0.),
                 ns_padding=(0., 0., 0.)):
        self.mode = mode
        self.scale = scale
        self.keep_in_memory = keep_in_memory

        self.ps_padding = ps_padding
        self.ns_padding = ns_padding

    def preload(self, space: Space):

        # Stage 1. Initialization of FileSources
        for node in PostOrderIter(space):
            if isinstance(node, FileSource):
                self._init_FileSource(node)

        # Stage 2. Initialization of Object3D
        for node in PostOrderIter(space):
            if isinstance(node, Object3D):
                self._init_Object3D(node)

                # Unload data from memory if it is necessary
                if not self.keep_in_memory:
                    for node2 in PreOrderIter(space):
                        if isinstance(node2, FileSource):
                            node2._loader.unload_data()

        # Stage 3. Initialization of Group
        for node in PostOrderIter(space):
            if isinstance(node, Group):
                self._init_Group(node)

        self._init_Space(space)
        self._keep_alphabetical_order_of_nodes(space)

    def _keep_alphabetical_order_of_nodes(self, space: Space):
        for node in PreOrderIter(space):
            node._NodeMixin__children_or_empty.sort(key=lambda d: (100 - DICT_NODETYPE_PRIORITY[d._nodetype], d.name),
                                                    reverse=False)

    def _add_sampling_node(self, node: AbstractBBoxNode):
        SamplingMethodSetNode(parent=node)

    def _init_Space(self, node: Space):
        _update_bbox_bottom_to_up(node)
        self._add_sampling_node(node)
        node.init()

    def _init_Group(self, node: Group):
        _update_bbox_bottom_to_up(node)
        self._add_sampling_node(node)

    def _process_sdt_source(self, node: Object3D):

        sdt_array_list = [source for source in node.children
                          if isinstance(source, FileSource) and source.loader_type == 'sdt']
        transform = None
        if len(sdt_array_list) > 0:
            from nndt.space2.transformation import IdentityTransform, ShiftAndScaleTransform, ToNormalCubeTransform
            ps_bbox = sdt_array_list[0].bbox
            if self.mode == 'identity':
                transform = IdentityTransform(ps_bbox=ps_bbox,
                                              parent=node)
            elif self.mode == 'shift_and_scale':
                ps_center = ((ps_bbox[0][0] + ps_bbox[1][0]) / 2.,
                             (ps_bbox[0][1] + ps_bbox[1][1]) / 2.,
                             (ps_bbox[0][2] + ps_bbox[1][2]) / 2.)
                transform = ShiftAndScaleTransform(ps_bbox=ps_bbox,
                                                   ps_center=ps_center,
                                                   ns_center=(0., 0., 0.),
                                                   scale_ps2ns=self.scale,
                                                   parent=node)
            elif self.mode == 'to_cube':
                transform = ToNormalCubeTransform(ps_bbox=ps_bbox,
                                                  parent=node)
            else:
                raise NotImplementedError(f"{self.mode} is not supported for initialization")
            node.bbox = update_bbox(node.bbox, transform.bbox)
            node.bbox = pad_bbox(node.bbox, self.ns_padding)

            if len(sdt_array_list) and transform is not None:
                sdt = sdt_array_list[0]
                SDTMethodSetNode(node, sdt, transform, parent=node)
                TrainTaskSetNode(node, sdt, transform, parent=node)

        return transform

    def _process_mesh_obj_source(self, node: Object3D, transform: AbstractTransformation):
        mesh_obj_array_list = [source for source in node.children
                               if isinstance(source, FileSource) and source.loader_type == 'mesh_obj']

        if len(mesh_obj_array_list) and transform is not None:
            mesh = mesh_obj_array_list[0]
            MeshObjMethodSetNode(node, mesh, transform, parent=node)
            if mesh._loader.rgba is not None:
                ColorMethodSetNode(node, mesh, transform, parent=node)

    def _init_Object3D(self, node: Object3D):
        transform = self._process_sdt_source(node)
        self._process_mesh_obj_source(node, transform)
        self._add_sampling_node(node)

    def _init_FileSource(self, node: FileSource):

        from nndt.space2 import DICT_LOADERTYPE_CLASS
        if node.loader_type not in DICT_LOADERTYPE_CLASS:
            raise NotImplementedError(f'{node.loader_type} is unknown loader')

        node._loader = DICT_LOADERTYPE_CLASS[node.loader_type](filepath=node.filepath)
        node._loader.load_data()
        node.bbox = node._loader.calc_bbox()
        node.bbox = pad_bbox(node.bbox, self.ps_padding)

        self._add_sampling_node(node)

        if not self.keep_in_memory:
            node._loader.unload_data()
