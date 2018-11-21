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

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    return skel


# оргигинальное изображение
originalImage = cv2.imread('./fingers/101_2.tif', 0)
# бинаризируем изображение
binaryImg = cv2.adaptiveThreshold(originalImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# нормализуем изображение с помощью вычисления гистограмы
hist = cv2.calcHist([binaryImg], [0], None, [256], [0, 256])
# поиск главных компонент методом PCA
mean, eigenVectors = cv2.PCACompute(hist, np.mean(hist, 0))
print(mean, eigenVectors)
# улучшение изображения с помощью фильтра Габора
g_kernel = cv2.getGaborKernel((11, 11), 8.0, np.pi/4, 10.0, 0.5, 0)
filtered_image = cv2.filter2D(binaryImg, cv2.CV_8UC3, g_kernel)

skelet = skeletization(filtered_image)
cv2.imshow('gabor kernel (resized)', skelet)
cv2.waitKey(0)
cv2.destroyAllWindows()

images = [originalImage, binaryImg, skelet, filtered_image]
titles = ['Оригинал', 'Бинарное', 'skel', 'gabor']
viewResults(len(images), images, titles)
