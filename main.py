import cv2
import matplotlib.pyplot as plt
import numpy as np


def viewResults(count, images, titles):
    for i in range(count):
        plt.subplot(2,2,i+1),plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

def skeletization(img):
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 4))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
    return skel

originalImage = cv2.imread('./fingers/101_2.tif', 0)
binaryImg = cv2.adaptiveThreshold(originalImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

skelet = skeletization(binaryImg);

images = [originalImage, binaryImg, skelet]
titles = ['Оригинал', 'Бинарное', 'skel']
viewResults(len(images), images, titles)
