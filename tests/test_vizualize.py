import os
import shutil
import unittest

import numpy as np

from nndt.vizualize import BasicVizualization
from tests.base import BaseTestCase

LOG_FOLDER = "test_log"
EXP_NAME = "test_exp"
EPOCHS = 100


class VizualizeTestCase(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def tearDown(self) -> None:
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

    def test_sdf_to_obj_array_greater_than_level(self):
        viz = BasicVizualization(LOG_FOLDER, EXP_NAME, print_on_each_epoch=100)
        test_box = np.zeros((100, 100, 100))
        test_box[20:80, 20:80, 20:80] = 1
        viz.sdt_to_obj("state", test_box, level=-1)

        with open(f"./{LOG_FOLDER}/state.obj", "r") as fl:
            lines = fl.readlines()
        self.assertTrue(not lines)

    def test_sdf_to_obj_array_smaller_than_level(self):
        viz = BasicVizualization(LOG_FOLDER, EXP_NAME, print_on_each_epoch=100)
        test_box = np.zeros((100, 100, 100))
        test_box[20:80, 20:80, 20:80] = 1
        viz.sdt_to_obj("state", test_box, level=1000)

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

    def test_train(self):
        viz = BasicVizualization(LOG_FOLDER, EXP_NAME, print_on_each_epoch=10)
        for epoch in viz.iter(100):
            viz.record({"loss": float(0)})
            self.assertEqual((epoch % 10 == 0), viz.is_print_on_epoch(epoch))
            if viz.is_print_on_epoch(epoch):
                viz.draw_loss("TRAIN_LOSS", viz._records["loss"])
                self.assertTrue(os.path.exists(f"./{LOG_FOLDER}/TRAIN_LOSS.jpg"))


if __name__ == "__main__":
    unittest.main()
