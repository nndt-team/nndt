import os
import pickle
import time
from typing import *

import jax.numpy as jnp
import matplotlib.pylab as plt
import numpy as onp

from nndt.space.repr_mesh import SaveMesh
from nndt.space2 import array_to_vert_and_faces, save_verts_and_faces_to_obj


class IteratorWithTimeMeasurements:
    """Iterator for recording of time and epoch number"""

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


class BasicVizualization:
    """
    Simple MLOps class for store train history and visualization of intermediate results
    """

    def __init__(self, folder: str,
                 experiment_name: Optional[str] = None,
                 print_on_each_epoch=20):
        """
        Simple MLOps class for store train history and visualization of intermediate results

        :param folder: folder for store results
        :param experiment_name: name for an experiments
        :param print_on_each_epoch: this parameter helps to control intermediate result output
        """
        self.folder = folder
        self.experiment_name = experiment_name if (experiment_name is not None) else folder
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
        """Check if this is the epoch for print results

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
        """Saves the training history in .jpg

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
        pickle.dump(state, open(os.path.join(self.folder, f"{name}.pkl"), 'wb'))

    def save_txt(self, name, summary):
        """Saves string data to .txt file

        Parameters
        ----------
        name : string
            File name
        summary : string
            The text to save
        """
        with open(os.path.join(self.folder, f"{name}.txt"), 'w') as fl:
            fl.write(summary)

    def sdt_to_obj(self, filename: str,
                   array: Union[jnp.ndarray, onp.ndarray],
                   level: float = 0.):
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
        assert (array.ndim == 3)
        array_ = onp.array(array)

        verts, faces = array_to_vert_and_faces(array_, level=level)
        save_verts_and_faces_to_obj(os.path.join(self.folder, f"{filename}.obj"), verts, faces)

    def save_mesh(self, name, save_method: SaveMesh, dict_):
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
        """Saves 3D array to file and section of this array as images

        Parameters
        ----------
        name : string
            File name
        array : array
            3D array to save
        section_img : bool
            If true, this saves three plane section of 3D array (defaults to True)
        """
        assert (array.ndim == 3)
        jnp.save(os.path.join(self.folder, f"{name}.npy"), array)

        if section_img:
            plt.close(1)
            plt.figure(1)
            plt.title(f"{self.experiment_name}_{name}_0")
            plt.imshow(array[array.shape[0] // 2, :, :], cmap='turbo')
            plt.colorbar()
            plt.savefig(os.path.join(self.folder, f"{name}_0.jpg"))

            plt.close(1)
            plt.figure(1)
            plt.title(f"{self.experiment_name}_{name}_1")
            plt.imshow(array[:, array.shape[1] // 2, :], cmap='turbo')
            plt.colorbar()
            plt.savefig(os.path.join(self.folder, f"{name}_1.jpg"))

            plt.close(1)
            plt.figure(1)
            plt.title(f"{self.experiment_name}_{name}_2")
            plt.imshow(array[:, :, array.shape[2] // 2], cmap='turbo')
            plt.colorbar()
            plt.savefig(os.path.join(self.folder, f"{name}_2.jpg"))
