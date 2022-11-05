__version__ = "0.0.3rc4"

from nndt.datagen import DataGenForSegmentation, DataGenForShapeRegression
from nndt.global_config import PYVISTA_PRE_PARAMS, init_code, init_colab, init_jupyter
from nndt.haiku_modules import DescConv, LipLinear, LipMLP
from nndt.math_core import (
    barycentric_grid,
    grid_in_cube,
    grid_in_cube2,
    help_barycentric_grid,
    rotation_matrix,
    scale_xyz,
    shift_xyz,
    take_each_n,
    train_test_split,
    uniform_in_cube,
)
from nndt.primitive_sdf import SphereSDF, fun2vec_and_grad
from nndt.trainable_task import (
    ApproximateSDF,
    ApproximateSDFLipMLP,
    Eikonal3D,
    SimpleSDF,
    SurfaceSegmentation,
)
from nndt.vizualize import BasicVizualization, save_3D_slices, save_sdt_as_obj
