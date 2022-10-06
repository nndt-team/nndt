import os.path
import unittest
import io
import inspect

from nndt.space2 import *

FILE_TMP = "./test_file.space"
FILE_TMP2 = "./test_file2.space"


class SpaceModelBeforeInitializationTestCase(unittest.TestCase):

    def setUp(self) -> None:
        if os.path.exists(FILE_TMP):
            os.remove(FILE_TMP)
        if os.path.exists(FILE_TMP2):
            os.remove(FILE_TMP2)

    def test_load_from_path(self):
        space = load_from_path("./test_folder_tree")
        print(space.explore())

    def test_space_modes(self):
        space = load_from_path("./test_folder_tree")
        print(text1 := space.explore('default'))
        print(text2 := space.explore('full'))
        self.assertLessEqual(len(text1), len(text2))

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

    def test_save_space_and_load_space(self):
        space = load_from_path("./acdc_for_test")
        print(text1 := space.explore())
        space.save_space(FILE_TMP)
        space2 = load_space(FILE_TMP)
        print(text2 := space2.explore())
        self.assertEqual(text1, text2)
        space2.save_space(FILE_TMP2)
        with open(FILE_TMP, 'r') as fl:
            with open(FILE_TMP2, 'r') as fl2:
                self.assertEqual(fl.readlines(), fl2.readlines())

        space = load_from_path("./test_folder_tree")
        print(text1 := space.explore())
        space.save_space(FILE_TMP)
        space2 = load_space(FILE_TMP)
        print(text2 := space2.explore())
        self.assertEqual(text1, text2)
        space2.save_space(FILE_TMP2)
        with open(FILE_TMP, 'r') as fl:
            with open(FILE_TMP2, 'r') as fl2:
                self.assertEqual(fl.readlines(), fl2.readlines())

    def test_to_json_and_from_json(self):
        space = load_from_path("./acdc_for_test")
        print(text1 := space.explore())
        json1 = space.to_json()
        space2 = from_json(json1)
        print(text2 := space2.explore())
        self.assertEqual(text1, text2)
        json2 = space2.to_json()
        self.assertEqual(json1, json2)

        space = load_from_path("./test_folder_tree")
        print(text1 := space.explore())
        json1 = space.to_json()
        space2 = from_json(json1)
        print(text2 := space2.explore())
        self.assertEqual(text1, text2)
        json2 = space2.to_json()
        self.assertEqual(json1, json2)

    def test_node_decorator(self):
        space = load_from_path("./acdc_for_test")


        print(space.explore())


        # def get_decorators(function):
        #     """Returns list of decorators names
        #
        #     Args:
        #         function (Callable): decorated method/function
        #
        #     Return:
        #         List of decorators as strings
        #
        #     Example:
        #         Given:
        #
        #         @my_decorator
        #         @another_decorator
        #         def decorated_function():
        #             pass
        #
        #         >>> get_decorators(decorated_function)
        #         ['@my_decorator', '@another_decorator']
        #
        #     """
        #     source = inspect.getsource(function)
        #     index = source.find("def ")
        #     return [
        #         line.strip().split()[0]
        #         for line in source[:index].strip().splitlines()
        #         if line.strip()[0] == "@"
        #     ]




if __name__ == '__main__':
    unittest.main()
