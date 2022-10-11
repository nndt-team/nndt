import unittest

import jax.numpy as jnp

from nndt.space2 import load_from_path

FILE_TMP = "./test_file.space"
FILE_TMP2 = "./test_file2.space"

PATH_TEST_STRUCTURE = './test_folder_tree'
PATH_TEST_ACDC = './acdc_for_test'


class LoadersTestCase(unittest.TestCase):

    def cmp_array(self, arr0, arr1, atol=0.1):
        arr0_ = jnp.array(arr0)
        arr1_ = jnp.array(arr1)
        print(arr1_)
        self.assertTrue(bool(jnp.allclose(arr0_, arr1_, atol=atol, rtol=0.0)))
        return arr1_

    def helper_preload_call(self, path, keep_in_memory):
        space = load_from_path(path)
        space.preload(keep_in_memory=keep_in_memory)
        print(space.print('default'))
        print(space.print('full'))

        return space

    def test_preload_call(self):
        self.helper_preload_call(PATH_TEST_ACDC, keep_in_memory=False)
        self.helper_preload_call(PATH_TEST_STRUCTURE, keep_in_memory=False)

    def test_preload_call_and_keep_in_memory(self):
        self.helper_preload_call(PATH_TEST_ACDC, keep_in_memory=True)
        self.helper_preload_call(PATH_TEST_STRUCTURE, keep_in_memory=True)

    def test_preload_check_access_to_field_mesh(self):
        space = self.helper_preload_call(PATH_TEST_ACDC, keep_in_memory=False)

        self.assertNotIn('^', space.patient069.colored_obj.print())
        self.assertIsNotNone(space.patient069.colored_obj._loader.mesh)
        self.assertIn('^', space.patient069.colored_obj.print())

    def BROKEN_test_preload_check_access_to_field_text(self):
        space = self.helper_preload_call(PATH_TEST_STRUCTURE, keep_in_memory=False)

        self.assertNotIn('^', space.group1.patient11.organ110.data1100_txt.print())
        self.assertIsNotNone(space.group1.patient11.organ110.data1100_txt._loader.text)
        self.assertIn('^', space.group1.patient11.organ110.data1100_txt.print())

    def test_preload_ps_bbox_in_mesh(self):
        space = self.helper_preload_call(PATH_TEST_ACDC, keep_in_memory=False)
        tolerance = 0.00001

        self.cmp_array(((58.81672286987305, 111.76485443115234, 6.518979072570801),
                        (133.49337768554688, 182.3278045654297, 83.50870513916016)),
                       space.patient009.colored_obj.bbox, tolerance)
        self.cmp_array(((44.65834426879883, 120.142333984375, 6.617307186126709),
                        (119.0753173828125, 197.04644775390625, 89.1721420288086)),
                       space.patient029.colored_obj.bbox, tolerance)
        self.cmp_array(((98.19243621826172, 77.09327697753906, 11.006719589233398),
                        (173.4209747314453, 149.07855224609375, 60.4520378112793)),
                       space.patient049.colored_obj.bbox, tolerance)
        self.cmp_array(((68.10173034667969, 108.21791076660156, 7.120966911315918),
                        (125.36803436279297, 155.4543914794922, 54.33668899536133)),
                       space.patient069.colored_obj.bbox, tolerance)
        self.cmp_array(((90.06974029541016, 105.29344177246094, 6.555858135223389),
                        (162.07835388183594, 166.13941955566406, 50.982513427734375)),
                       space.patient089.colored_obj.bbox, tolerance)

    def test_preload_ps_bbox_in_sdt(self):
        space = self.helper_preload_call(PATH_TEST_ACDC, keep_in_memory=False)
        tolerance = 1.1

        self.cmp_array(((58.81672286987305, 111.76485443115234, 6.518979072570801),
                        (133.49337768554688, 182.3278045654297, 83.50870513916016)),
                       space.patient009.sdf_npy.bbox, tolerance)
        self.cmp_array(((44.65834426879883, 120.142333984375, 6.617307186126709),
                        (119.0753173828125, 197.04644775390625, 89.1721420288086)),
                       space.patient029.sdf_npy.bbox, tolerance)
        self.cmp_array(((98.19243621826172, 77.09327697753906, 11.006719589233398),
                        (173.4209747314453, 149.07855224609375, 60.4520378112793)),
                       space.patient049.sdf_npy.bbox, tolerance)
        self.cmp_array(((68.10173034667969, 108.21791076660156, 7.120966911315918),
                        (125.36803436279297, 155.4543914794922, 54.33668899536133)),
                       space.patient069.sdf_npy.bbox, tolerance)
        self.cmp_array(((90.06974029541016, 105.29344177246094, 6.555858135223389),
                        (162.07835388183594, 166.13941955566406, 50.982513427734375)),
                       space.patient089.sdf_npy.bbox, tolerance)

    def helper_transform_load(self, mode="identity"):
        space = load_from_path(PATH_TEST_ACDC)
        space.preload(mode=mode, keep_in_memory=False)
        print(space.print("full"))

        return space

    def test_preload_identity(self):
        space = self.helper_transform_load(mode="identity")

        self.cmp_array(((59.0, 112.0, 7.0), (134.0, 183.0, 84.0)), space.patient009.sdf_npy.bbox)
        self.cmp_array(((59.0, 112.0, 7.0), (134.0, 183.0, 84.0)), space.patient009.transform.bbox)
        self.cmp_array(((0.0, 0.0, 0.0), (134.0, 183.0, 84.0)), space.patient009.bbox)
        self.cmp_array(((0.0, 0.0, 0.0), (174.0, 198.0, 90.0)), space.bbox)

    def test_preload_shift_and_scale(self):
        space = self.helper_transform_load(mode="shift_and_scale")

        self.cmp_array(((59.0, 112.0, 7.0), (134.0, 183.0, 84.0)), space.patient009.sdf_npy.bbox)
        self.cmp_array(((-0.75, -0.71, -0.77), (0.75, 0.71, 0.77)), space.patient009.transform.bbox)
        self.cmp_array(((-0.75, -0.71, -0.77), (0.75, 0.71, 0.77)), space.patient009.bbox)
        self.cmp_array(((-0.76, -0.78, -0.83), (0.76, 0.78, 0.83)), space.bbox)

    def test_preload_to_cube(self):
        space = self.helper_transform_load(mode="to_cube")

        self.cmp_array(((59.0, 112.0, 7.0), (134.0, 183.0, 84.0)), space.patient009.sdf_npy.bbox)
        self.cmp_array(((-1, -1, -1), (1, 1, 1)), space.patient009.transform.bbox)
        self.cmp_array(((-1, -1, -1), (1, 1, 1)), space.patient009.bbox)
        self.cmp_array(((-1, -1, -1), (1, 1, 1)), space.bbox)

    def test_request_tree_nodes2(self):
        space = self.helper_transform_load(mode="identity")

        self.assertEqual(5, len(space))
        self.assertEqual(2, len(space.patient009))

        obj1 = space.patient009.colored_obj
        obj2 = space[0][0]
        obj3 = space['patient009']['colored_obj']
        obj4 = space['patient009/colored_obj']

        self.assertEqual(obj1, obj2)
        self.assertEqual(obj1, obj3)
        self.assertEqual(obj1, obj4)

if __name__ == '__main__':
    unittest.main()
