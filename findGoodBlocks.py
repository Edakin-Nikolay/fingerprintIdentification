import cv2
import numpy as np
from math import atan2

# константы
TEST_BLOCK_H = 1
TEST_BLOCK_W = 2

MIN_POINTS = 2
THRESHOLD = 50

IMG_WIDTH = 480
BLOCK_WIDTH = 240
IMG_HEIGHT = 640
BLOCK_HEIGHT = 160
# углы поворота в ГРАДУСАХ
ANGLES = [0, 23, 45, 68, 90, 113, 135, 158]


def pcaAngle(img):
    indicesY, indicesX = np.where(img < THRESHOLD)
    points = np.column_stack((indicesX, indicesY)).astype(np.float64)
    if points.shape[0] < MIN_POINTS:
        return 0.0

    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(points, mean)
    angle = np.pi/2 - atan2(eigenvectors[0, 1], eigenvectors[0, 0])
    # print(angle, np.rad2deg(angle))
    return angle


def gaborize(img, angle):
    # if points.shape[0] < MIN_POINTS:
    #    return img

    g_kernel = cv2.getGaborKernel((11, 11), 4.0, angle, 8.0, 0.5)
    # cv2.imshow('g_kernel', g_kernel)

    filtered_image = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
    return filtered_image



def splitImg2blocks(img, xs=0, ys=0):
    if xs == 0 & ys == 0:
        xs = range(IMG_WIDTH // BLOCK_WIDTH)
        ys = range(IMG_HEIGHT // BLOCK_HEIGHT)
    blocks = []
    for x in xs:
        for y in ys:
            imgY = y * BLOCK_HEIGHT
            imgX = x * BLOCK_WIDTH
            blocks.append(img[imgY:(imgY+BLOCK_HEIGHT), imgX:(imgX+BLOCK_WIDTH)])
    return blocks


def combineBlockAngle(pair):
    blocks, angle = pair
    return list(map(lambda block: [block, angle], blocks))


def calcGoodBlock(listPairs):
    absAngle = list(map(lambda pair: [pair[0], abs(pair[1] - pcaAngle(pair[0]))], listPairs))
    return min(absAngle, key=lambda pair: pair[1])


# оргигинальное изображение
originalImage = cv2.imread('./fingers/101_1.tif', cv2.IMREAD_GRAYSCALE)

equalized = cv2.equalizeHist(originalImage)
gaborized_blocks = []

# создаём набор блоков каждого изображения на каждый угол
for angle in ANGLES:
    img = cv2.bitwise_not(gaborize(equalized, np.deg2rad(angle)))
    blocks = splitImg2blocks(img)
    gaborized_blocks.append([blocks, np.deg2rad(angle)])

zipped_blocks = list(zip(*list(map(combineBlockAngle, gaborized_blocks))))
blocksAndAngle = list(map(calcGoodBlock, zipped_blocks))
goodBlocks = list(map(lambda pair: pair[0], blocksAndAngle))

print(len(goodBlocks))
cv2.imshow('block', goodBlocks[2])
equalized2 = equalized
""""
for x in range(2):
    for y in range(4):
        imgY = y * BLOCK_HEIGHT
        imgX = x * BLOCK_WIDTH
        print('equalized', len(equalized2[imgY:(imgY+BLOCK_HEIGHT), imgX:(imgX+BLOCK_WIDTH)]))
        print('block', len(goodBlocks[x + y][0]))
        equalized2[imgY:(imgY+BLOCK_HEIGHT), imgX:(imgX+BLOCK_WIDTH)] = goodBlocks[x + y][0]
"""
# cv2.imshow('equalized', equalized2)
# cv2.imshow('equalized', equalized2[0:BLOCK_HEIGHT, 0:BLOCK_WIDTH])
# print('block', len(goodBlocks[0][0]))
# cv2.imshow('block', goodBlocks[0][0])
# cv2.imshow('res', equalized2)
# отдельный блок для тестов
"""
block = equalized[blockHeight*TEST_BLOCK_H:(blockHeight*TEST_BLOCK_H+blockHeight), blockWidth*TEST_BLOCK_W:(blockWidth*TEST_BLOCK_W+blockWidth)]
gaborize_block = gaborize(block)
cv2.imshow('block', block)
cv2.imshow('block_gabor', gaborize_block)
equalized = cv2.rectangle(equalized, (TEST_BLOCK_W*blockWidth, TEST_BLOCK_H*blockHeight), ((TEST_BLOCK_W+1)*blockWidth, (TEST_BLOCK_H+1)*blockHeight), (255, 0, 0), 1)
cv2.imshow('original equalize', equalized)
"""

# цикл разбивки изображения на части и применения фильтра
"""
equalized2 = equalized
for x in xs:
    for y in ys:
        imgY = y * blockHeight
        imgX = x * blockWidth
        equalized2[imgY:(imgY+blockHeight), imgX:(imgX+blockWidth)] = gaborize(equalized[imgY:(imgY+blockHeight), imgX:(imgX+blockWidth)])

# бинаризируем изображение
# binaryImg = cv2.adaptiveThreshold(cv2.bitwise_not(img), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow('equalized2', equalized2)
"""
cv2.waitKey()
