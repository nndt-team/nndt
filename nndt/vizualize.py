import os
import pickle
import time

import jax.numpy as jnp
import matplotlib.pylab as plt
import numpy as onp
from skimage import measure

from nndt.space.repr_mesh import SaveMesh


class IteratorWithTimeMeasurements:
    """Iter on BasicVizualization object and shows work duration"""

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

    def __init__(self, folder, experiment_name=None, print_on_each_epoch=20):
        self.folder = folder
        self.experiment_name = experiment_name if (experiment_name is not None) else folder
        os.makedirs(self.folder, exist_ok=True)
        self.print_on_each_epoch = print_on_each_epoch
        self._records = {"_epoch": [], "_time": []}

    def iter(self, epochs_num):
        """_summary_

        Parameters
        ----------
        epochs_num : int
            number of epochs
                      
        Returns
        -------
            IteratorWithTimeMeasurements
        """
        return IteratorWithTimeMeasurements(self, epochs_num)

    def record(self, dict):
        for k, v in dict.items():
            if k in self._records:
                self._records[k].append(v)
            else:
                self._records[k] = []
                self._records[k].append(v)

    def is_print_on_epoch(self, epoch):
        """Returns the epoch to print

        Parameters
        ----------
        epoch : int
            epoch number

        Returns
        -------
        bool 
            should we print it?
        """
        return (epoch % self.print_on_each_epoch) == 0

    def draw_loss(self, name, history):
        """Saves a loss in jpg

        Parameters
        ----------
        name : string
            filename
        history (_type_): 
            loss history
        """
        plt.close(1)
        plt.figure(1)
        plt.semilogy(history)
        plt.title(f"{self.experiment_name}_{name}")
        plt.grid()
        plt.savefig(os.path.join(self.folder, f"{name}.jpg"))

    def save_state(self, name, state):
        """Writes a serialized view of state in file

        Parameters
        ----------
        name : string 
            filename
        state : (_type_)
            state to save
        """
        pickle.dump(state, open(os.path.join(self.folder, f"{name}.pkl"), 'wb'))

    def save_txt(self, name, summary):
        """Saves txt to file

        Parameters
        ----------
        name : string
            filename
        summary : string
            text to save
        """
        with open(os.path.join(self.folder, f"{name}.txt"), 'w') as fl:
            fl.write(summary)

    def sdf_to_obj(self, name, array, level=0.):
        """Converts sdf to obj

        Parameters
        ----------
        name : string 
            filename
        array : array
            array of dim 3
        level : (_type_, optional)
            _description_. (defaults to 0.)
        """
        assert (array.ndim == 3)
        array_ = onp.array(array)

        level_ = level
        if not (array_.min() < level_ < array_.max()):
            level_ = (array_.max() + array_.min()) / 2.

        verts, faces, _, _ = measure.marching_cubes(array_, level=level_)

        with open(os.path.join(self.folder, f"{name}.obj"), 'w') as fl:
            for v in verts:
                fl.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for f in faces:
                fl.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")

    def save_mesh(self, name, save_method: SaveMesh, dict_):
        """Saves mesh

        Parameters
        ----------
        name : string
            filename
        save_method : SaveMesh 
            save method is SaveMesh(file_path: str, name_value: dict)
        dict_ : dict
            name_value
        """
        save_method(os.path.join(self.folder, f"{name}.vtp"), dict_)

    def save_3D_array(self, name, array, section_img=True):
        """Saves 3D array

        Parameters
        ----------
        name : string
            filename
        array : array
            3D array to save
        section_img : bool
            add an image (defaults to True)
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
