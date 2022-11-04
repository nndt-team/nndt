from typing import Tuple

PYVISTA_PRE_PARAMS = dict()


def init_colab(window_size: Tuple[int, int] = (600, 400)):
    """
    This method rewrites global parameters and allows NNDT works inside the Google Colaboratory environment

    :param window_size: default size of output 3D images
    :return: none
    """
    import os

    os.system("/usr/bin/Xvfb :99 -screen 0 1024x768x24 &")
    os.environ["DISPLAY"] = ":99"

    import panel as pn
    import pyvista as pv

    pn.extension("vtk")

    PYVISTA_PRE_PARAMS["notebook"] = True
    PYVISTA_PRE_PARAMS["window_size"] = window_size


def init_jupyter():
    """
    This method rewrites global parameters and allows NNDT works inside the jupyter notebook environment

    :return: none
    """
    pass


def init_code():
    """
    This method rewrites global parameters and allows NNDT works inside the Jupyter notebook environment

    :return: none
    """
    pass
