import unittest
import findGoodBlocks as fgb
import cv2
import numpy as np


class TestFindGoodBlocks(unittest.TestCase):
    def setUp(self):
        self.inputImg = cv2.equalizeHist(cv2.imread('./fingers/101_1.tif', cv2.IMREAD_GRAYSCALE))
        self.inputArray = np.array([np.array([0, 1, 2, 3]),
                                   np.array([4, 5, 6, 7]),
                                   np.array([8, 9, 10, 11]),
                                   np.array([12, 13, 14, 15]),
                                   np.array([16, 17, 18, 19]),
                                   np.array([20, 21, 22, 23])])
        self.angles = [0, 30, 45, 60, 90, 120]
        self.blocksAngle = \
            list(map(lambda angle: [fgb.splitImg2blocks(self.inputArray, xs=2, ys=3, imgWidth=4, imgHeight=6), angle], self.angles))


    def test_up(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_splitImg2blocks(self):
        splitted = fgb.splitImg2blocks(self.inputArray, xs=2, ys=3, imgWidth=4, imgHeight=6)
        self.assertEqual(len(splitted), 6)
        self.assertEqual(splitted[2][0][0], 16)

    def test_combineBlocksAndAngles(self):
        self.assertEqual(self.blocksAngle[2][1], 45)
        self.assertEqual(self.blocksAngle[2][0][2][0][0], 16)

    def test_combineBlockAngle(self):
        res = list(map(fgb.combineBlockAngle, self.blocksAngle))
        self.assertEqual(len(res), 6)
        self.assertEqual(res[3][3][1], 60)
        self.assertEqual(res[3][2][0][0][0], 16)

    def test_zipBlocks(self):
        listZip = list(zip(*list(map(fgb.combineBlockAngle, self.blocksAngle))))
        self.assertEqual(len(listZip[5]), 6)
        # print(listZip[4])

    def test_cutAngles(self):
        listZip = list(zip(*list(map(fgb.combineBlockAngle, self.blocksAngle))))
        withoutAngles = list(map(fgb.cutAngles, listZip))
        self.assertEqual(withoutAngles[0][0][1][0], 4)

    def test_makeImage(self):
        listZip = list(zip(*list(map(fgb.combineBlockAngle, self.blocksAngle))))
        withoutAngles = list(map(fgb.cutAngles, listZip))
        goodBlocks = [withoutAngles[0][0], withoutAngles[1][1], withoutAngles[2][2],
                      withoutAngles[3][3], withoutAngles[4][4], withoutAngles[5][5]]
        # print(fgb.makeImage(goodBlocks, 3))
        self.assertEqual(fgb.makeImage(goodBlocks, 3), self.inputArray)


if __name__ == '__main__':
    unittest.main()
