import os
import unittest

from nndt.space2 import *
from tests.base import PATH_TEST_ACDC, BaseTestCase

FILE_TMP_PNG = "./test_file.png"


class MyTestCase(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def setUp(self) -> None:
        if os.path.exists(FILE_TMP_PNG):
            os.remove(FILE_TMP_PNG)

    def test_plot_colored_obj__identity_transform(self):
        space = load_from_path(PATH_TEST_ACDC)
        space.preload(mode="identity", keep_in_memory=False)
        print(space.print("default"))

        space.patient009.colored_obj.plot(filepath=FILE_TMP_PNG)
        self.assertTrue(os.path.exists(FILE_TMP_PNG))

    def test_plot_sdt_npy__identity_transform(self):
        space = load_from_path(PATH_TEST_ACDC)
        space.preload(mode="identity", keep_in_memory=False)
        print(space.print("default"))

        space.patient009.sdf_npy.plot(filepath=FILE_TMP_PNG)
        self.assertTrue(os.path.exists(FILE_TMP_PNG))

    def test_plot_object3D__identity_transform(self):
        space = load_from_path(PATH_TEST_ACDC)
        space.preload(mode="identity", keep_in_memory=False)
        print(space.print("default"))

        space.patient009.plot(filepath=FILE_TMP_PNG)
        self.assertTrue(os.path.exists(FILE_TMP_PNG))

    def test_plot_Space__identity_transform(self):
        space = load_from_path(PATH_TEST_ACDC)
        space.preload(mode="identity", keep_in_memory=False)
        print(space.print("default"))

        space.plot(filepath=FILE_TMP_PNG)
        self.assertTrue(os.path.exists(FILE_TMP_PNG))

    def test_plot_Space__shift_and_scale(self):
        space = load_from_path(PATH_TEST_ACDC)
        space.preload(mode="shift_and_scale", keep_in_memory=False)
        print(space.print("default"))

        space.plot(filepath=FILE_TMP_PNG)
        self.assertTrue(os.path.exists(FILE_TMP_PNG))

    def test_plot_Space__to_cube(self):
        space = load_from_path(PATH_TEST_ACDC)
        space.preload(mode="to_cube", keep_in_memory=False)
        print(space.print("default"))

        space.plot(filepath=FILE_TMP_PNG)
        self.assertTrue(os.path.exists(FILE_TMP_PNG))

    def test_plot_before_preload(self):
        with self.assertWarns(Warning):
            space = load_from_path(PATH_TEST_ACDC)
            space.plot(filepath=FILE_TMP_PNG)
        self.assertTrue(os.path.exists(FILE_TMP_PNG))

    def test_plot_windows_size(self):
        with self.assertWarns(Warning):
            space = load_from_path(PATH_TEST_ACDC)
            space.plot(filepath=FILE_TMP_PNG)
        self.assertTrue(os.path.exists(FILE_TMP_PNG))

    def notest_for_manual_run(self):
        with self.assertWarns(Warning):
            space = load_from_path(PATH_TEST_ACDC)
            space.preload(mode="identity", keep_in_memory=False)
            space.plot()

    def notest_for_manual_run2(self):
        with self.assertWarns(Warning):
            space = load_from_path(PATH_TEST_ACDC)
            space.preload(mode="identity", keep_in_memory=False)
            space.patient009.plot(window_size=(700, 700), cmap="Set1")

    def test_plot_tree(self):
        space = Space("space")
        space = add_sphere(space, "sp1", center=(1.0, 1.0, 1.0), radius=1.0)
        space = add_sphere(space, "sp2", center=(-1.0, -1.0, -1.0), radius=2.0)
        space.preload()
        print(space.print("default"))
        # space.plot()

        space.plot(filepath=FILE_TMP_PNG)
        self.assertTrue(os.path.exists(FILE_TMP_PNG))

    def test_plot_Space__to_cube__add_sphere(self):
        space = load_from_path(PATH_TEST_ACDC)
        space.preload(mode="to_cube", keep_in_memory=False)
        space = add_sphere(space, "sp1", center=(0.0, -0.0, -0.0), radius=1.0)
        print(space.print("default"))
        # space.plot()

        space.plot(filepath=FILE_TMP_PNG)
        self.assertTrue(os.path.exists(FILE_TMP_PNG))

    def test_plot_Space__shift_and_scale__add_sphere(self):
        space = load_from_path(PATH_TEST_ACDC)
        space.preload(mode="shift_and_scale", keep_in_memory=False)
        space = add_sphere(space, "sp1", center=(0.0, 0.0, 0.0), radius=1.0)
        print(space.print("default"))
        # space.plot()

        space.plot(filepath=FILE_TMP_PNG)
        self.assertTrue(os.path.exists(FILE_TMP_PNG))

    def test_plot_Space__identity_transform__add_sphere(self):
        space = load_from_path(PATH_TEST_ACDC)
        space.preload(mode="identity", keep_in_memory=False)
        space = add_sphere(space, "sp1", center=(0.0, 0.0, 0.0), radius=1.0)
        space = add_sphere(space, "sp1", center=(10.0, 10.0, 10.0), radius=5.0)
        print(space.print("default"))
        # space.plot()

        space.plot(filepath=FILE_TMP_PNG)
        self.assertTrue(os.path.exists(FILE_TMP_PNG))


if __name__ == "__main__":
    unittest.main()
