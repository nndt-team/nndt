import unittest

from nndt.space2 import *


class MyTestCase(unittest.TestCase):

    def test_load_from_path(self):
        space = load_from_path("./test_folder_tree")
        print(space.explore())

    def test_load_from_path2(self):
        space = load_from_path("./acdc_for_test")
        print(space.explore())

    def test_forbidden_name(self):
        with self.assertRaises(ValueError) as context:
            space = Space('children')

    def test_request_tree_nodes(self):
        space = load_from_path("./acdc_for_test")

        obj1 = space.patient069
        obj2 = space[0]
        obj3 = space['patient069']

        self.assertEqual(obj1, obj2)
        self.assertEqual(obj1, obj3)

    def test_request_tree_nodes2(self):
        space = load_from_path("./acdc_for_test")

        obj1 = space.patient069.colored_obj
        obj2 = space[0][0]
        obj3 = space['patient069']['colored_obj']
        obj4 = space['patient069/colored_obj']

        self.assertEqual(obj1, obj2)
        self.assertEqual(obj1, obj3)
        self.assertEqual(obj1, obj4)

    def test_explore(self):
        space = load_from_path("./acdc_for_test")

        print(text1 := space.patient069.explore())
        print(text2 := space[0].explore())
        print(text3 := space['patient069'].explore())

        self.assertEqual(text1, text2)
        self.assertEqual(text1, text3)


if __name__ == '__main__':
    unittest.main()
