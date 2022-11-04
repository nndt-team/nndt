import os.path
import unittest

import jax

from nndt.space2 import *
from nndt.space2.space import Space
from nndt.space2.utils import update_bbox
from tests.base import PATH_TEST_ACDC, PATH_TEST_STRUCTURE, BaseTestCase

FILE_TMP = "./test_file.space"
FILE_TMP2 = "./test_file2.space"


class SpaceModelBeforeInitializationTestCase(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def setUp(self) -> None:
        if os.path.exists(FILE_TMP):
            os.remove(FILE_TMP)
        if os.path.exists(FILE_TMP2):
            os.remove(FILE_TMP2)

    def test_load_from_path(self):
        space = load_from_path(PATH_TEST_STRUCTURE)
        space.init()
        text1 = space.print().__str__()
        text2 = space.print().__repr__()
        self.assertEqual(text1, text2)
        print(space.print())

    def test_no_double_init(self):
        space = load_from_path(PATH_TEST_STRUCTURE)
        space.init()
        text1 = space.print()
        print(text1)
        space.init()
        text2 = space.print()
        print(text2)
        self.assertEqual(text1, text2)

        space = load_from_path(PATH_TEST_ACDC)
        space.init()
        text1 = space.print()
        print(text1)
        space.init()
        text2 = space.print()
        print(text2)
        self.assertEqual(text1, text2)

    def test_space_models_TEST_STRUCTURE(self):
        space = load_from_path(PATH_TEST_STRUCTURE)
        space.init()
        text1 = space.print("source")
        print(text1)
        text2 = space.print("default")
        print(text2)
        text3 = space.print("full")
        print(text3)
        self.assertLessEqual(len(text1), len(text2))
        self.assertLessEqual(len(text2), len(text3))

    def test_space_models_ACDC(self):
        space = load_from_path(PATH_TEST_ACDC)
        space.init()
        text1 = space.print("source")
        print(text1)
        text2 = space.print("default")
        print(text2)
        text3 = space.print("full")
        print(text3)
        self.assertLessEqual(len(text1), len(text2))
        self.assertLessEqual(len(text2), len(text3))

    def test_load_from_path2(self):
        space = load_from_path(PATH_TEST_ACDC)
        space.init()
        print(space.print())

    def test_forbidden_name(self):
        space = Space("children")
        space.init()
        self.assertEqual("children_", space.name)

    def test_request_tree_nodes(self):
        space = load_from_path(PATH_TEST_ACDC)
        space.init()

        obj1 = space.patient009
        obj2 = space[0]
        obj3 = space["patient009"]

        self.assertEqual(obj1, obj2)
        self.assertEqual(obj1, obj3)

    def test_request_tree_nodes2(self):
        space = load_from_path(PATH_TEST_ACDC)
        space.init()

        self.assertEqual(5, len(space))
        self.assertEqual(2, len(space.patient009))

        obj1 = space.patient009.colored_obj
        obj2 = space[0][0]
        obj3 = space["patient009"]["colored_obj"]
        obj4 = space["patient009/colored_obj"]

        self.assertEqual(obj1, obj2)
        self.assertEqual(obj1, obj3)
        self.assertEqual(obj1, obj4)

    def test_print(self):
        space = load_from_path(PATH_TEST_ACDC)
        space.init()

        text1 = space.patient009.print()
        print(text1)
        text2 = space[0].print()
        print(text2)
        text3 = space["patient009"].print()
        print(text3)

        self.assertEqual(text1, text2)
        self.assertEqual(text1, text3)

    def helper_save_space_and_load_space(self, pathname):
        space = load_from_path(pathname)
        space.init()
        text1 = space.print()
        print(text1)
        space.save_space_to_file(FILE_TMP)
        space2 = read_space_from_file(FILE_TMP)
        space2.init()
        text2 = space2.print()
        print(text2)
        self.assertEqual(text1, text2)
        space2.save_space_to_file(FILE_TMP2)
        with open(FILE_TMP, "r") as fl:
            with open(FILE_TMP2, "r") as fl2:
                self.assertEqual(fl.readlines(), fl2.readlines())

    def test_save_space_and_load_space(self):
        self.helper_save_space_and_load_space(PATH_TEST_ACDC)
        self.helper_save_space_and_load_space(PATH_TEST_STRUCTURE)

    def helper_to_json_and_from_json(self, pathname):
        space = load_from_path(pathname)
        space.init()
        text1 = space.print()
        print(text1)
        json1 = space.to_json()
        space2 = from_json(json1)
        space2.init()
        text2 = space2.print()
        print(text2)
        self.assertEqual(text1, text2)
        json2 = space2.to_json()
        self.assertEqual(json1, json2)

    def test_to_json_and_from_json(self):
        self.helper_to_json_and_from_json(PATH_TEST_ACDC)
        self.helper_to_json_and_from_json(PATH_TEST_STRUCTURE)

    def test_node_decorator(self):
        space = load_from_path(PATH_TEST_ACDC)
        space.init()
        print(space.print("full"))
        self.assertTrue(hasattr(space, "patient009"))
        self.assertTrue(hasattr(space, "patient029"))
        self.assertTrue(hasattr(space, "patient049"))
        self.assertTrue(hasattr(space, "patient069"))
        self.assertTrue(hasattr(space, "patient089"))
        self.assertIn("print", [x.name for x in space.children])
        self.assertIn("patient009", [x.name for x in space.children])
        self.assertIn("patient029", [x.name for x in space.children])
        self.assertIn("patient049", [x.name for x in space.children])
        self.assertIn("patient069", [x.name for x in space.children])
        self.assertIn("patient089", [x.name for x in space.children])

    def test_load_txt(self):
        space = load_txt(
            PATH_TEST_STRUCTURE + "/group0/patient00/organ000/data0000.txt"
        )
        print(space.print("full"))
        self.assertEqual(1, len(space))
        self.assertEqual(1, len(space.default))
        space.preload()

    def test_load_sdt(self):
        space = load_sdt(PATH_TEST_ACDC + "/patient089/sdf.npy")
        print(space.print("full"))
        self.assertEqual(1, len(space))
        self.assertEqual(1, len(space.default))
        space.preload()

    def test_load_mesh_obj(self):
        space = load_mesh_obj(PATH_TEST_ACDC + "/patient089/colored.obj")
        print(space.print("full"))
        self.assertEqual(1, len(space))
        self.assertEqual(1, len(space.default))
        space.preload()

    def test_load_from_path_None_in_template(self):
        space = load_from_path(PATH_TEST_ACDC, template_sdt=None)
        tree = space.print("full")
        print(tree)
        self.assertNotIn("sdf_npy", tree)
        self.assertIn("colored_obj", tree)

        space = load_from_path(PATH_TEST_ACDC, template_mesh_obj=None)
        tree = space.print("full")
        print(tree)
        self.assertIn("sdf_npy", tree)
        self.assertNotIn("colored_obj", tree)

        space = load_from_path(
            PATH_TEST_ACDC, template_sdt=None, template_mesh_obj=None
        )
        tree = space.print("full")
        print(tree)
        self.assertNotIn("sdf_npy", tree)
        self.assertNotIn("colored_obj", tree)

    def test_preload_empty(self):
        space = load_from_path(
            PATH_TEST_ACDC, template_sdt=None, template_mesh_obj=None
        )
        space.preload()
        print(space.print("full"))

    def test_test_test_train_split(self):
        rng_key = jax.random.PRNGKey(42)

        space = load_from_path(PATH_TEST_ACDC)
        space.preload()
        print(space.print("default"))
        self.assertEqual(5, len(space))
        space = split_node_test_train(rng_key, space, test_size=0.4)
        self.assertEqual(2, len(space.test))
        self.assertEqual(3, len(space.train))
        print(space.print("default"))
        self.assertEqual(space.bbox, update_bbox(space.test.bbox, space.train.bbox))

    def helper_load_kfold(self):
        space = load_from_path(PATH_TEST_ACDC)
        space.preload()
        return space

    def test_split_node_kfold(self):
        space = self.helper_load_kfold()
        space = split_node_kfold(space, n_fold=5, k_for_test=0)
        print(space.print("source"))
        _ = space.test.patient009
        _ = space.train.patient029
        _ = space.train.patient049
        _ = space.train.patient069
        _ = space.train.patient089

    def test_split_node_kfold2(self):
        space = self.helper_load_kfold()
        space = split_node_kfold(space, n_fold=5, k_for_test=4)
        print(space.print("source"))
        _ = space.train.patient009
        _ = space.train.patient029
        _ = space.train.patient049
        _ = space.train.patient069
        _ = space.test.patient089

    def test_split_node_kfold_list(self):
        space = self.helper_load_kfold()
        space = split_node_kfold(space, n_fold=5, k_for_test=[0, 1, 4])
        print(space.print("source"))
        _ = space.test.patient009
        _ = space.test.patient029
        _ = space.train.patient049
        _ = space.train.patient069
        _ = space.test.patient089

    def test_split_node_kfold_protection(self):

        space = self.helper_load_kfold()
        with self.assertRaises(ValueError):
            space = split_node_kfold(space, n_fold=6, k_for_test=3)

        space = self.helper_load_kfold()
        with self.assertRaises(ValueError):
            space = split_node_kfold(space, n_fold=5, k_for_test=6)

        space = self.helper_load_kfold()
        with self.assertRaises(ValueError):
            space = split_node_kfold(space, n_fold=6, k_for_test=[0, 1, 2])

        space = self.helper_load_kfold()
        with self.assertRaises(ValueError):
            space = split_node_kfold(space, n_fold=5, k_for_test=[0, 1, 1])

        space = self.helper_load_kfold()
        with self.assertRaises(ValueError):
            space = split_node_kfold(space, n_fold=5, k_for_test=[0, 1, 6])

    def test_split_node_namelist(self):
        space = self.helper_load_kfold()
        space = split_node_namelist(
            space,
            {
                "part01": ["patient009", "patient029"],
                "part02": ["patient049", "patient069"],
                "part03": ["patient089"],
            },
        )
        print(space.print("source"))
        _ = space.part01.patient009
        _ = space.part01.patient029
        _ = space.part02.patient049
        _ = space.part02.patient069
        _ = space.part03.patient089

    def test_split_node_namelist_value_errors(self):
        space = self.helper_load_kfold()
        with self.assertRaises(ValueError):
            space = split_node_namelist(
                space,
                {
                    "part01": ["patient009", "patient009"],
                    "part02": ["patient049", "patient069"],
                    "part03": ["patient089"],
                },
            )
        space = self.helper_load_kfold()
        with self.assertRaises(ValueError):
            space = split_node_namelist(
                space,
                {
                    "part01": ["patient009", "patient029"],
                    "part02": ["patient049", "patient069"],
                    "part03": [],
                },
            )

    def test_helper_in_node_method(self):

        space = load_from_path(PATH_TEST_ACDC)
        space.preload()
        self.assertLess(300, len(space.print.__doc__))

    def test_preload_with_and_without_verbose(self):
        space = load_from_path(PATH_TEST_ACDC)
        space.preload(verbose=False)

        space = load_from_path(PATH_TEST_ACDC)
        space.preload(verbose=True)

    def test_preload_check_condition(self):
        space = load_from_path(PATH_TEST_ACDC)
        self.assertFalse(space._is_preload)
        space.preload(verbose=False)
        self.assertTrue(space._is_preload)


if __name__ == "__main__":
    unittest.main()
