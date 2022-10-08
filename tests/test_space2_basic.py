import os.path
import unittest

from nndt.space2 import *

FILE_TMP = "./test_file.space"
FILE_TMP2 = "./test_file2.space"

PATH_TEST_STRUCTURE = './test_folder_tree'
PATH_TEST_ACDC = './acdc_for_test'


class SpaceModelBeforeInitializationTestCase(unittest.TestCase):

    def setUp(self) -> None:
        if os.path.exists(FILE_TMP):
            os.remove(FILE_TMP)
        if os.path.exists(FILE_TMP2):
            os.remove(FILE_TMP2)

    def test_load_from_path(self):
        space = load_from_path(PATH_TEST_STRUCTURE)
        print(space.explore())

    def test_space_models_TEST_STRUCTURE(self):
        space = load_from_path(PATH_TEST_STRUCTURE)
        print(text1 := space.explore('source'))
        print(text2 := space.explore('default'))
        print(text3 := space.explore('full'))
        self.assertLessEqual(len(text1), len(text2))
        self.assertLessEqual(len(text2), len(text3))

    def test_space_models_ACDC(self):
        space = load_from_path(PATH_TEST_ACDC)
        space.initialization()
        print(text1 := space.explore('source'))
        print(text2 := space.explore('default'))
        print(text3 := space.explore('full'))
        self.assertLessEqual(len(text1), len(text2))
        self.assertLessEqual(len(text2), len(text3))

    def test_load_from_path2(self):
        space = load_from_path(PATH_TEST_ACDC)
        print(space.explore())

    def test_forbidden_name(self):
        with self.assertRaises(ValueError) as context:
            space = Space('children')

    def test_request_tree_nodes(self):
        space = load_from_path(PATH_TEST_ACDC)

        obj1 = space.patient069
        obj2 = space[0]
        obj3 = space['patient069']

        self.assertEqual(obj1, obj2)
        self.assertEqual(obj1, obj3)

    def test_request_tree_nodes2(self):
        space = load_from_path(PATH_TEST_ACDC)

        obj1 = space.patient069.colored_obj
        obj2 = space[0][0]
        obj3 = space['patient069']['colored_obj']
        obj4 = space['patient069/colored_obj']

        self.assertEqual(obj1, obj2)
        self.assertEqual(obj1, obj3)
        self.assertEqual(obj1, obj4)

    def test_explore(self):
        space = load_from_path(PATH_TEST_ACDC)

        print(text1 := space.patient069.explore())
        print(text2 := space[0].explore())
        print(text3 := space['patient069'].explore())

        self.assertEqual(text1, text2)
        self.assertEqual(text1, text3)

    def helper_save_space_and_load_space(self, pathname):
        space = load_from_path(pathname)
        print(text1 := space.explore())
        space.save_space(FILE_TMP)
        space2 = load_space(FILE_TMP)
        print(text2 := space2.explore())
        self.assertEqual(text1, text2)
        space2.save_space(FILE_TMP2)
        with open(FILE_TMP, 'r') as fl:
            with open(FILE_TMP2, 'r') as fl2:
                self.assertEqual(fl.readlines(), fl2.readlines())

    def test_save_space_and_load_space(self):
        self.helper_save_space_and_load_space(PATH_TEST_ACDC)
        self.helper_save_space_and_load_space(PATH_TEST_STRUCTURE)

    def helper_to_json_and_from_json(self, pathname):
        space = load_from_path(pathname)
        print(text1 := space.explore())
        json1 = space.to_json()
        space2 = from_json(json1)
        print(text2 := space2.explore())
        self.assertEqual(text1, text2)
        json2 = space2.to_json()
        self.assertEqual(json1, json2)

    def test_to_json_and_from_json(self):
        self.helper_to_json_and_from_json(PATH_TEST_ACDC)
        self.helper_to_json_and_from_json(PATH_TEST_STRUCTURE)

    def test_node_decorator(self):
        space = load_from_path(PATH_TEST_ACDC)
        print(space.explore('full'))
        space.do_nothing()
        self.assertTrue(hasattr(space, 'patient009'))
        self.assertTrue(hasattr(space, 'patient029'))
        self.assertTrue(hasattr(space, 'patient049'))
        self.assertTrue(hasattr(space, 'patient069'))
        self.assertTrue(hasattr(space, 'patient089'))
        self.assertIn('do_nothing', [x.name for x in space.children])
        self.assertIn('explore', [x.name for x in space.children])
        self.assertIn('patient009', [x.name for x in space.children])
        self.assertIn('patient029', [x.name for x in space.children])
        self.assertIn('patient049', [x.name for x in space.children])
        self.assertIn('patient069', [x.name for x in space.children])
        self.assertIn('patient089', [x.name for x in space.children])


if __name__ == '__main__':
    unittest.main()
