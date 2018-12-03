import unittest
import findGoodBlocks as fgb
import cv2
import numpy as np


class TestFindGoodBlocks(unittest.TestCase):
    def setUp(self):
        self.inputImg = cv2.equalizeHist(cv2.imread('./fingers/101_1.tif', cv2.IMREAD_GRAYSCALE))
        self.inputArray = np.array([np.array([0, 0, 0, 0]),
                                   np.array([0, 0, 0, 0]),
                                   np.array([0, 0, 0, 0]),
                                   np.array([0, 0, 0, 0]),
                                   np.array([0, 0, 0, 0]),
                                   np.array([0, 0, 0, 0])])


    def test_up(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_splitImg2blocks(self):
        self.assertEqual(len(fgb.splitImg2blocks(self.inputImg)), 8)


if __name__ == '__main__':
    unittest.main()
