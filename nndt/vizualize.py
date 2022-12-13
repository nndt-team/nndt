import os
import pickle
import time
from typing import *

import jax.numpy as jnp
import matplotlib.pylab as plt
import numpy as np
import numpy as onp
from mpl_toolkits.axes_grid1 import AxesGrid

from nndt.space2.utils import fix_file_extension


class IteratorWithTimeMeasurements:
    """Iterator that records and prints the epoch number and time spent from the start of iterations"""

    def __init__(self, basic_viz, epochs):
        self.basic_viz = basic_viz
        self.time_start = time.time()
        self.time_previous = self.time_start
        self.epochs = epochs
        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.basic_viz.record({"_epoch": self.counter})

        time_full = time.time() - self.time_start
        self.basic_viz.record({"_time": time_full})

        if self.basic_viz.is_print_on_epoch(self.counter):
            str_ = f"[E:{self.basic_viz._records['_epoch'][-1]},T:{self.basic_viz._records['_time'][-1]:.01f}] "
            for k, v in self.basic_viz._records.items():
                if not k.startswith("_"):
                    str_ = str_ + f"{k}={v[-1]}, "
            str_ = str_ + "\n"
            print(str_)

        if self.counter > self.epochs:
            raise StopIteration()
        self.counter += 1

        return self.counter - 1

    def __len__(self):
        return self.epochs


def save_sdt_as_obj(
    array: Union[jnp.ndarray, onp.ndarray], path: str, level: float = 0.0
):
    """
    Run marching cubes over SDT and save results to file

    Parameters
    ----------
    filename : string
        File name
    array : ndarray
        Signed distance tensor (SDT)
    level : float
        Isosurface level (defaults to 0.).
    """
    assert array.ndim == 3
    array_ = onp.array(array)

    from nndt.space2.utils import array_to_vert_and_faces, save_verts_and_faces_to_obj

    verts, faces = array_to_vert_and_faces(array_, level=level)
    save_verts_and_faces_to_obj(fix_file_extension(path, ".obj"), verts, faces)


def save_3D_slices(
    array: Union[onp.ndarray, jnp.ndarray],
    path: str = None,
    slice_num: int = 5,
    include_boundary=True,
    figsize=None,
    levels=(0.0,),
    level_colors=("white",),
    **kwargs,
):
    """
    Generates a panel of images with slices of the 3D array. This is a helper function for studying 3D tensors.

    :param path: path to the image for write
    :param array: studied array
    :param slice_num: number of slices over array axis
    :param include_boundary: If True, the image will include boundaries of the array with indexes 0 and len(array)-1
    :param figsize: the size of the image. If None, the size will be calculated according to the number of panels.
    :param levels: Isoline values. This param is ignored if RGB/RGBA image is passed.
    :param level_colors: Isoline colors. This param is ignored if RGB/RGBA image is passed.
    :param kwargs: parameter set that is passed to the `.imshow()` method
    :return: none
    """
    panel_size = slice_num if include_boundary else slice_num - 2
    assert panel_size > 0
    assert slice_num > 0 and panel_size > 0
    assert array.ndim == 3 or (array.ndim == 4 and (array.shape[-1] in (1, 3, 4)))
    assert len(levels) == len(level_colors)

    is_color = array.ndim == 4 and (array.shape[-1] in (3, 4))

    if figsize is None:
        figsize = (3 * panel_size, 3 * 3)

    fig = plt.figure(figsize=figsize)
    if is_color:
        grid = AxesGrid(
            fig,
            111,
            nrows_ncols=(3, panel_size),
            axes_pad=0.05,
            label_mode="L",
        )
    else:
        grid = AxesGrid(
            fig,
            111,
            nrows_ncols=(3, panel_size),
            axes_pad=0.05,
            cbar_mode="single",
            cbar_location="right",
            cbar_pad=0.1,
            label_mode="L",
        )

    slices_x = np.linspace(0, array.shape[0] - 1, slice_num).astype(int)
    slices_y = np.linspace(0, array.shape[1] - 1, slice_num).astype(int)
    slices_z = np.linspace(0, array.shape[2] - 1, slice_num).astype(int)

    if not include_boundary:
        slices_x = slices_x[1:-1]
        slices_y = slices_y[1:-1]
        slices_z = slices_z[1:-1]

    array_ = array[..., np.newaxis] if (array.ndim == 3) else array
    for ind_ax, ax in zip(slices_x, grid[:panel_size]):
        im = ax.imshow(
            array_[ind_ax, :, :, 0:3].squeeze(),
            vmin=float(jnp.nanmin(array_)),
            vmax=float(jnp.nanmax(array_)),
            **kwargs,
        )
        if not is_color and len(levels) > 0 and len(level_colors) > 0:
            _cs2 = ax.contour(
                array_[ind_ax, :, :, 0:3].squeeze(),
                levels=levels,
                origin="lower",
                colors=level_colors,
            )
    for ind_ax, ax in zip(slices_y, grid[panel_size : panel_size * 2]):
        im = ax.imshow(
            array_[:, ind_ax, :, 0:3].squeeze(),
            vmin=float(jnp.nanmin(array_)),
            vmax=float(jnp.nanmax(array_)),
            **kwargs,
        )
        if not is_color and len(levels) > 0 and len(level_colors) > 0:
            _cs2 = ax.contour(
                array_[:, ind_ax, :, 0:3].squeeze(),
                levels=levels,
                origin="lower",
                colors=level_colors,
            )
    for ind_ax, ax in zip(slices_z, grid[2 * panel_size :]):
        im = ax.imshow(
            array_[:, :, ind_ax, 0:3].squeeze(),
            vmin=float(jnp.nanmin(array_)),
            vmax=float(jnp.nanmax(array_)),
            **kwargs,
        )
        if not is_color and len(levels) > 0 and len(level_colors) > 0:
            _cs2 = ax.contour(
                array_[:, :, ind_ax, 0:3].squeeze(),
                levels=levels,
                origin="lower",
                colors=level_colors,
            )

    if not is_color:
        cbar = ax.cax.colorbar(im)
        cbar = grid.cbar_axes[0].colorbar(im)
        if not is_color and len(levels) > 0 and len(level_colors) > 0:
            cbar.add_lines(_cs2)

    if path is not None:
        fig.savefig(path)
    else:
        plt.show()


class BasicVizualization:
    """
    Simple MLOps class for storing the train history and visualization of intermediate results
    """

    def __init__(
        self, folder: str, experiment_name: Optional[str] = None, print_on_each_epoch=20
    ):
        """
        Simple MLOps class for storing the train history and visualization of intermediate results

        :param folder: folder for store results
        :param experiment_name: name for an experiments
        :param print_on_each_epoch: this parameter helps to control intermediate result output
        """
        self.folder = folder
        self.experiment_name = (
            experiment_name if (experiment_name is not None) else folder
        )
        os.makedirs(self.folder, exist_ok=True)
        self.print_on_each_epoch = print_on_each_epoch
        self._records = {"_epoch": [], "_time": []}

    def iter(self, epoch_num):
        """Return iterators for the main train cycle

        Parameters
        ----------
        epoch_num : int
            number of epoch

        Returns
        -------
            instance of IteratorWithTimeMeasurements
        """
        return IteratorWithTimeMeasurements(self, epoch_num)

    def record(self, dict):
        for k, v in dict.items():
            if k in self._records:
                self._records[k].append(v)
            else:
                self._records[k] = []
                self._records[k].append(v)

    def is_print_on_epoch(self, epoch):
        """Check if this is the right epoch to print results

        Parameters
        ----------
        epoch : int
            epoch number

        Returns
        -------
        bool
            Should we print on this step?
        """
        return (epoch % self.print_on_each_epoch) == 0

    def draw_loss(self, name, history):
        """Save the training history in .jpg

        Parameters
        ----------
        name : string
            File name
        history (_type_):
            List of loss values over epochs
        """
        plt.close(1)
        plt.figure(1)
        plt.semilogy(history)
        plt.title(f"{self.experiment_name}_{name}")
        plt.grid()
        plt.savefig(os.path.join(self.folder, f"{name}.jpg"))

    def save_state(self, name, state):
        """Save the neural network state into the file

        Parameters
        ----------
        name : string
            File name
        state : (_type_)
            The state to save
        """
        pickle.dump(state, open(os.path.join(self.folder, f"{name}.pkl"), "wb"))

    def save_txt(self, name, summary):
        """Save string data to .txt file

        Parameters
        ----------
        name : string
            File name
        summary : string
            The text to save
        """
        with open(os.path.join(self.folder, f"{name}.txt"), "w") as fl:
            fl.write(summary)

    def sdt_to_obj(
        self, filename: str, array: Union[jnp.ndarray, onp.ndarray], level: float = 0.0
    ):
        """Run marching cubes over SDT and save results to file

        Parameters
        ----------
        filename : string
            File name
        array : ndarray
            Signed distance tensor (SDT)
        level : float
            Isosurface level (defaults to 0.).
        """
        save_sdt_as_obj(array, os.path.join(self.folder, f"{filename}.obj"), level)

    def save_mesh(self, name, save_method, dict_):
        """Save mesh to .vtp file with data

        Parameters
        ----------
        name : string
            filename
        save_method : SaveMesh
            SaveMesh instance from NNDT space (v0.0.1 or v0.0.2)
        dict_ : dict
            name_value
        """
        save_method(os.path.join(self.folder, f"{name}.vtp"), dict_)

    def save_3D_array(self, name, array, section_img=True):
        """Save the 3D array to a file and section of this array as images

        Parameters
        ----------
        name : string
            File name
        array : array
            3D array to save
        section_img : bool
            If true, this saves three plane section of 3D array (defaults to True)
        """
        assert array.ndim == 3
        jnp.save(os.path.join(self.folder, f"{name}.npy"), array)

        if section_img:
            plt.close(1)
            plt.figure(1)
            plt.title(f"{self.experiment_name}_{name}_0")
            plt.imshow(array[array.shape[0] // 2, :, :])
            plt.colorbar()
            plt.savefig(os.path.join(self.folder, f"{name}_0.jpg"))

            plt.close(1)
            plt.figure(1)
            plt.title(f"{self.experiment_name}_{name}_1")
            plt.imshow(array[:, array.shape[1] // 2, :])
            plt.colorbar()
            plt.savefig(os.path.join(self.folder, f"{name}_1.jpg"))

            plt.close(1)
            plt.figure(1)
            plt.title(f"{self.experiment_name}_{name}_2")
            plt.imshow(array[:, :, array.shape[2] // 2])
            plt.colorbar()
            plt.savefig(os.path.join(self.folder, f"{name}_2.jpg"))


class ANSIConverter:
    """
    Create a converter from ANSI to RGB color
    :param value: value that is recieved using Fore.VALUE or Back.VALUE from colorama
    :param type: type of ANSI code (use 'Fore' or 'Back' value)
    """

    def __init__(self, value: str, type: str):
        self.ANSIcode = value
        self.type = type,

        self.__RGB_color = {
            '30': 'rgb(1,1,1)',
            '31': 'rgb(222,56,43)',
            '32': 'rgb(57,181,74)',
            '33': 'rgb(255,199,6)',
            '34': 'rgb(0,111,184)',
            '35': 'rgb(118,38,113)',
            '36': 'rgb(44,181,233)',
            '37': 'rgb(204,204,204)',

            '90': 'rgb(128,128,128)',
            '91': 'rgb(255,0,0)',
            '92': 'rgb(0,255,0)',
            '93': 'rgb(255,255,0)',
            '94': 'rgb(0,0,255)',
            '95': 'rgb(255,0,255)',
            '96': 'rgb(0,255,255)',
            '97': 'rgb(255,255,255)',
        }

    def get_str_code(self):
        from re import search

        return search("\d{2,}", self.ANSIcode)[0]

    def is_types_equals(self, str_code: str):
        is_equals = False
        code = int(str_code)

        match self.type[0]:
            case 'Fore':
                is_equals = code in range(30, 38) or code in range(90, 98)
            case 'Back':
                is_equals = code in range(30, 48) or code in range(100, 108)

        return is_equals

    def to_rgb(self):
        str_code = self.get_str_code()

        assert self.is_types_equals(str_code)

        match self.type[0]:
            case 'Fore':
                return self.__RGB_color[str_code]
            case 'Back':
                return self.__RGB_color[str(int(str_code) - 10)]

