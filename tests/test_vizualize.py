import os
import shutil
import unittest

import numpy as np

from nndt.space2.utils import fix_file_extension
from nndt.vizualize import BasicVizualization, save_3D_slices
from tests.base import BaseTestCase

LOG_FOLDER = "test_log"
EXP_NAME = "test_exp"
EPOCHS = 100


class VizualizeTestCase(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def tearDown(self) -> None:
        if os.path.exists(f"./{LOG_FOLDER}"):
            shutil.rmtree(f"./{LOG_FOLDER}")

    def test_draw_loss(self):
        viz = BasicVizualization(LOG_FOLDER, EXP_NAME, print_on_each_epoch=100)
        viz.draw_loss("state", [1, 2, 1, 2, 1, 2, 1, 2])
        self.assertTrue(os.path.exists(f"./{LOG_FOLDER}/state.jpg"))

    def test_save_state(self):
        viz = BasicVizualization(LOG_FOLDER, EXP_NAME, print_on_each_epoch=100)
        viz.save_state("state", "state")
        self.assertTrue(os.path.exists(f"./{LOG_FOLDER}/state.pkl"))

    def test_save_txt(self):
        viz = BasicVizualization(LOG_FOLDER, EXP_NAME, print_on_each_epoch=100)
        viz.save_txt("state", "state")
        self.assertTrue(os.path.exists(f"./{LOG_FOLDER}/state.txt"))

    def test_sdf_to_obj(self):
        viz = BasicVizualization(LOG_FOLDER, EXP_NAME, print_on_each_epoch=100)
        test_box = np.zeros((100, 100, 100))
        test_box[20:80, 20:80, 20:80] = 1
        viz.sdt_to_obj("state", test_box, level=0.5)
        self.assertTrue(os.path.exists(f"./{LOG_FOLDER}/state.obj"))

    def test_sdf_to_obj_calc_level(self):
        viz = BasicVizualization(LOG_FOLDER, EXP_NAME, print_on_each_epoch=100)
        test_box = np.zeros((100, 100, 100))
        test_box[20:80, 20:80, 20:80] = 1
        with self.assertWarns(Warning):
            viz.sdt_to_obj("state", test_box, level=-1)
        self.assertTrue(os.path.exists(f"./{LOG_FOLDER}/state.obj"))

    def test_sdf_to_obj_array_of_the_equal_values(self):
        viz = BasicVizualization(LOG_FOLDER, EXP_NAME, print_on_each_epoch=100)
        test_box = np.ones((100, 100, 100))
        with self.assertWarns(Warning):
            viz.sdt_to_obj("state", test_box, level=-1)
        with open(f"./{LOG_FOLDER}/state.obj", "r") as fl:
            lines = fl.readlines()
        self.assertTrue(not lines)

    def test_save_3D_array(self):
        viz = BasicVizualization(LOG_FOLDER, EXP_NAME, print_on_each_epoch=100)
        test_box = np.zeros((100, 100, 100))
        test_box[20:80, 20:80, 20:80] = 1
        viz.save_3D_array("state", test_box, section_img=True)
        self.assertTrue(os.path.exists(f"./{LOG_FOLDER}/state.npy"))
        self.assertTrue(os.path.exists(f"./{LOG_FOLDER}/state_0.jpg"))
        self.assertTrue(os.path.exists(f"./{LOG_FOLDER}/state_1.jpg"))
        self.assertTrue(os.path.exists(f"./{LOG_FOLDER}/state_2.jpg"))

    def test_save_3D_slices(self):
        test_box = np.zeros((100, 100, 100))
        test_box[20:80, 20:80, 20:80] = 1
        os.makedirs(f"./{LOG_FOLDER}/")
        save_3D_slices(test_box, f"./{LOG_FOLDER}/slices0.png")
        save_3D_slices(test_box, f"./{LOG_FOLDER}/slices1.png", include_boundary=False)
        self.assertTrue(os.path.exists(f"./{LOG_FOLDER}/slices0.png"))
        self.assertTrue(os.path.exists(f"./{LOG_FOLDER}/slices1.png"))

    def test_train(self):
        viz = BasicVizualization(LOG_FOLDER, EXP_NAME, print_on_each_epoch=10)
        for epoch in viz.iter(100):
            viz.record({"loss": float(0)})
            self.assertEqual((epoch % 10 == 0), viz.is_print_on_epoch(epoch))
            if viz.is_print_on_epoch(epoch):
                viz.draw_loss("TRAIN_LOSS", viz._records["loss"])
                self.assertTrue(os.path.exists(f"./{LOG_FOLDER}/TRAIN_LOSS.jpg"))

    def test_fix_file_extension(self):
        str = fix_file_extension("/something/file.txt", ".txt")
        self.assertTrue("/something/file.txt", str)
        str = fix_file_extension("/something/file.wow", ".txt")
        self.assertTrue("/something/file.wow.txt", str)
        str = fix_file_extension("/something/file", ".txt")
        self.assertTrue("/something/file.txt", str)
        str = fix_file_extension("/something/file.", ".txt")
        self.assertTrue("/something/file..txt", str)


if __name__ == "__main__":
    unittest.main()
