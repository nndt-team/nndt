from colorama import Fore

from nndt.space2.abstracts import AbstractBBoxNode
from nndt.space2.utils import update_bbox
from nndt.space2 import FileSource


class Object3D(AbstractBBoxNode):
    def __init__(self, name,
                 bbox=((0., 0., 0.), (0., 0., 0.)),
                 parent=None):
        super(Object3D, self).__init__(name, parent=parent, bbox=bbox, _print_color=Fore.BLUE, _nodetype='O3D')

    def __repr__(self):
        return self._print_color + f'{self._nodetype}:{self.name}' + Fore.WHITE + f' {self._print_bbox()}' + Fore.RESET

    # def _initialization(self, mode='ident', scale=50, keep_in_memory=False):
    #     sdt_array_list = [source for source in self.children
    #                       if isinstance(source, FileSource) and source.loader_type == 'sdt']
    #     transform = None
    #     if len(sdt_array_list):
    #         from nndt.space2.transformation import IdentityTransform, ShiftAndScaleTransform, ToNormalCubeTransform
    #         ps_bbox = sdt_array_list[0].bbox
    #         if mode == 'ident':
    #             transform = IdentityTransform(ps_bbox=ps_bbox,
    #                                           parent=self)
    #         elif mode == 'shift_and_scale':
    #             ps_center = ((ps_bbox[0][0] + ps_bbox[1][0]) / 2.,
    #                          (ps_bbox[0][1] + ps_bbox[1][1]) / 2.,
    #                          (ps_bbox[0][2] + ps_bbox[1][2]) / 2.)
    #             transform = ShiftAndScaleTransform(ps_bbox=ps_bbox,
    #                                                ps_center=ps_center,
    #                                                ns_center=(0., 0., 0.),
    #                                                scale_ps2ns=scale,
    #                                                parent=self)
    #         elif mode == 'to_cube':
    #             transform = ToNormalCubeTransform(ps_bbox=ps_bbox,
    #                                               parent=self)
    #         else:
    #             raise NotImplementedError(f"{mode} is not supported for initialization")
    #         self.bbox = update_bbox(self.bbox, transform.bbox)
    #
    #     mesh_obj_array_list = [source for source in self.children
    #                            if isinstance(source, FileSource) and source.loader_type == 'mesh_obj']
    #
    #     from nndt.space2 import MeshNode
    #     if len(mesh_obj_array_list) and transform is not None:
    #         mesh = mesh_obj_array_list[0]
    #         MeshNode(self, mesh, transform, parent=self)
    #
    #     from nndt.space2 import SamplingNode
    #     SamplingNode(parent=self)
