from typing import Tuple

PYVISTA_PRE_PARAMS = dict()


def init_colab(window_size: Tuple[int, int] = (600, 400)):
    import os

    os.system("/usr/bin/Xvfb :99 -screen 0 1024x768x24 &")
    os.environ["DISPLAY"] = ":99"

    import panel as pn
    import pyvista as pv

    pn.extension("vtk")

    PYVISTA_PRE_PARAMS = {
        "notebook": True,
        "window_size": window_size,
    }


def init_jupyter():
    pass


def init_code():
    pass
