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
                                   np.array([1, 0, 0, 0]),
                                   np.array([0, 0, 0, 0])])
        self.angles = [0, 30, 45, 60, 90, 120]


    def test_up(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_splitImg2blocks(self):
        splitted = fgb.splitImg2blocks(self.inputArray, xs=2, ys=3, imgWidth=4, imgHeight=6)
        self.assertEqual(len(splitted), 6)
        self.assertEqual(splitted[2][0][0], 1)

    def test_combineBlocksAndAngles(self):
        blocksAngle = []
        for angle in self.angles:
            blocks = fgb.splitImg2blocks(self.inputArray, xs=2, ys=3, imgWidth=4, imgHeight=6)
            blocksAngle.append([blocks, angle])
        self.assertEqual(blocksAngle[2][1], 45)
        self.assertEqual(blocksAngle[2][0][2][0][0], 1)

    def test_combineBlockAngle(self):
        blocksAngle = []
        for angle in self.angles:
            blocks = fgb.splitImg2blocks(self.inputArray, xs=2, ys=3, imgWidth=4, imgHeight=6)
            blocksAngle.append([blocks, angle])
        res = list(map(fgb.combineBlockAngle, blocksAngle))
        self.assertEqual(len(res), 6)
        self.assertEqual(res[3][3][1], 60)
        self.assertEqual(res[3][2][0][0][0], 1)
        listZip = list(zip(*res))
        self.assertEqual(len(listZip[0]), 6)


if __name__ == '__main__':
    unittest.main()
