__version__ = "0.0.3a2"

import nndt.space2.plot_tree
from nndt.math_core import *
from nndt.primitive_sdf import SphereSDF
from nndt.trainable_task import (
    ApproximateSDF,
    ApproximateSDFLipMLP,
    Eikonal3D,
    SimpleSDF,
    SurfaceSegmentation,
)
from nndt.vizualize import BasicVizualization


def init_colab():
    import os

    os.system("/usr/bin/Xvfb :99 -screen 0 1024x768x24 &")
    os.environ["DISPLAY"] = ":99"

    import numpy as np
    import panel as pn
    import pyvista as pv

    pn.extension("vtk")

    nndt.space2.plot_tree.pyvista_pre_params = {
        "notebook": True,
        "window_size": (600, 400),
    }
