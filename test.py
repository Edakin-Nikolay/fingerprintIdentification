import cv2
import numpy as np
from math import atan2

minPoints = 2
threshold = 25

def gaborize(img):
    indicesY, indicesX = np.where(img < threshold)
    points = np.column_stack((indicesX, indicesY)).astype(np.float64)

    if points.shape[0] < minPoints:
        return img

    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(points, mean)
    print(eigenvalues[0] > eigenvalues[1])

    angle = atan2(eigenvectors[0,1], eigenvectors[0,0])
    g_kernel = cv2.getGaborKernel((11, 11), 8.0, angle, 10.0, 0.5, 0)
    filtered_image = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
    return filtered_image

# оргигинальное изображение
originalImage = cv2.imread('./fingers/101_2.tif', 0)

imgWidth = 480
blockWidth = 16
xs = range(imgWidth // blockWidth)
imgHeight = 640
blockHeight = 18
ys = range(imgHeight // blockHeight)

equalized = cv2.equalizeHist(originalImage)
for x in xs:
    for y in ys:
        imgY = y * blockHeight
        imgX = x * blockWidth
        equalized[imgY:(imgY+blockHeight), imgX:(imgX+blockWidth)] = gaborize(equalized[imgY:(imgY+blockHeight), imgX:(imgX+blockWidth)])

# бинаризируем изображение
binaryImg = cv2.adaptiveThreshold(originalImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow('output', equalized)
cv2.waitKey()
