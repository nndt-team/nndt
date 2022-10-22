import unittest

from .base import BaseTestCase


class ImportTestCase(BaseTestCase):

    def test_math_core(self):
        from nndt import grid_in_cube, grid_in_cube2
        grid_in_cube()
        grid_in_cube2()

    def test_primitive_sdf(self):
        from nndt import SphereSDF
        SphereSDF()


if __name__ == '__main__':
    unittest.main()
