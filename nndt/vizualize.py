import os
import pickle
import time
from typing import *

import jax.numpy as jnp
import matplotlib.pylab as plt
import numpy as np
import numpy as onp

from nndt.space2 import fix_file_extension


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
    path: str, array: Union[jnp.ndarray, onp.ndarray], level: float = 0.0
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

    from nndt.space2 import array_to_vert_and_faces, save_verts_and_faces_to_obj

    verts, faces = array_to_vert_and_faces(array_, level=level)
    save_verts_and_faces_to_obj(fix_file_extension(path, ".obj"), verts, faces)


def save_3D_slices(
    path: str,
    array: Union[onp.ndarray, jnp.ndarray],
    slice_num: int = 5,
    include_boundary=True,
    figsize=None,
    **kwargs,
):
    """
    Generates panel of images with slices of 3D array. This is a helper function for studying 3D tensors.

    :param path: path to image for write
    :param array: studied array
    :param slice_num: number of slices over array axis
    :param include_boundary: If True, image will include boundaries of array with indexes 0 and len(array)-1
    :param figsize: size of image. If None, size will be calculated according to number of panels.
    :param kwargs: set of parameter that is passed to the `.imshow()` method
    :return: none
    """
    panel_size = slice_num if include_boundary else slice_num - 2
    assert panel_size > 0
    assert array.ndim == 3

    if figsize is None:
        figsize = (3 * panel_size, 3 * 3)

    fig, axs = plt.subplots(3, panel_size, figsize=figsize)
    slices_x = np.linspace(0, array.shape[0] - 1, slice_num).astype(int)
    slices_y = np.linspace(0, array.shape[1] - 1, slice_num).astype(int)
    slices_z = np.linspace(0, array.shape[2] - 1, slice_num).astype(int)

    if not include_boundary:
        slices_x = slices_x[1:-1]
        slices_y = slices_y[1:-1]
        slices_z = slices_z[1:-1]

    for ind_panel, ind_x in enumerate(slices_x):
        axs[0, ind_panel].imshow(array[ind_x, :, :], **kwargs)
        axs[0, ind_panel].set(ylabel="x", xlabel=str(ind_x))
    for ind_panel, ind_y in enumerate(slices_y):
        axs[1, ind_panel].imshow(array[:, ind_y, :], **kwargs)
        axs[1, ind_panel].set(ylabel="y", xlabel=str(ind_y))
    for ind_panel, ind_z in enumerate(slices_z):
        axs[2, ind_panel].imshow(array[:, :, ind_z], **kwargs)
        axs[2, ind_panel].set(ylabel="z", xlabel=str(ind_z))

    for ax in axs.flat:
        ax.label_outer()

    fig.tight_layout()
    fig.savefig(path)


class BasicVizualization:
    """
    Simple MLOps class for storing the train history and visualization of intermediate results
    """

    def __init__(
        self, folder: str, experiment_name: Optional[str] = None, print_on_each_epoch=20
    ):
        """
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
        """Save neural network state into the file

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
        save_sdt_as_obj(os.path.join(self.folder, f"{filename}.obj"), array, level)

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
        """Save 3D array to a file and section of this array as images

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
            plt.imshow(array[array.shape[0] // 2, :, :], cmap="turbo")
            plt.colorbar()
            plt.savefig(os.path.join(self.folder, f"{name}_0.jpg"))

            plt.close(1)
            plt.figure(1)
            plt.title(f"{self.experiment_name}_{name}_1")
            plt.imshow(array[:, array.shape[1] // 2, :], cmap="turbo")
            plt.colorbar()
            plt.savefig(os.path.join(self.folder, f"{name}_1.jpg"))

            plt.close(1)
            plt.figure(1)
            plt.title(f"{self.experiment_name}_{name}_2")
            plt.imshow(array[:, :, array.shape[2] // 2], cmap="turbo")
            plt.colorbar()
            plt.savefig(os.path.join(self.folder, f"{name}_2.jpg"))
