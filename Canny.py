import cv2 as cv
import numpy as np


def filter2D(image, kernel):
    height, width = image.shape[:2]
    result = np.zeros((height, width), dtype=np.int32)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            k = image[i - 1: i + 2, j - 1: j + 2]
            # print(k)
            r = np.sum(k * kernel)
            # print(r)
            result[i, j] = r

    return result


def nms(dst, angle):
    height, width = dst.shape[:2]
    result = np.zeros((height, width), dtype=np.int32)
    # angle = dst * 180. / np.pi
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            v1 = 255
            v2 = 255
            if angle[i, j] > 180:
                angle[i, j] = angle[i, j] - 180
            if 67.5 > angle[i, j] >= 22.5:
                # slash
                v1 = dst[i + 1, j - 1]
                v2 = dst[i - 1, j + 1]
            elif 112.5 > angle[i, j] >= 67.5:
                # horizontal
                v1 = dst[i + 1, j]
                v2 = dst[i - 1, j]
            elif 157.5 > angle[i, j] >= 112.5:
                # backslash
                v1 = dst[i - 1, j - 1]
                v2 = dst[i + 1, j + 1]
            elif 0 <= angle[i, j] < 22.5 or 157.5 <= angle[i, j] <= 180:
                # vertical
                v1 = dst[i, j + 1]
                v2 = dst[i, j - 1]
            # else:
            # print(angle[i, j])
            if dst[i, j] <= v1 or dst[i, j] <= v2:
                result[i, j] = 0
            else:
                result[i, j] = int(dst[i, j])
    return result


# 八連通元件
def eightConnectedComponent(x, y):
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    neighbors = [(x + dx, y + dy) for dx, dy in moves]
    return neighbors


def doubleThreshold(src, lowThreshold=50, highThreshold=150):
    height, width = src.shape[:2]
    result = np.zeros((height, width), dtype=np.uint8)
    binary = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if src[i, j] < lowThreshold:
                result[i, j] = 0
                binary[i, j] = 0
            elif lowThreshold <= src[i, j] < highThreshold:
                result[i, j] = 1  # weak
                binary[i, j] = 50
            else:
                result[i, j] = 2  # strong
                binary[i, j] = 255

    return result, binary


# 標記連通元件
def labelComponent(src, lowThreshold=50, highThreshold=150):
    thesholdingSrc, binarySrc = doubleThreshold(src, lowThreshold, highThreshold)
    # cv.imshow("binary", binarySrc)
    col, row = src.shape[0:2]
    labels = np.zeros((col, row), dtype=int)
    #     # print(labels.shape)
    result = np.array(src)
    #     # print(result.shape)
    # 用遞迴做標記，標記直到跳出連通元件
    for i in range(col):
        for j in range(row):
            stack = [(i, j)]
            if thesholdingSrc[i, j] > 1 and labels[i, j] == 0:
                while stack:
                    x, y = stack.pop()
                    if labels[x, y] == 0:
                        labels[x, y] = 1
                        neighbors = eightConnectedComponent(x, y)
                        # 只框出空間上有連結的點
                        for k, l in neighbors:
                            if 0 <= k < src.shape[1] and 0 <= l < src.shape[0] and thesholdingSrc[k, l] > 0:
                                stack.append((k, l))
    # 將元件邊緣保留並將元件以外的像素設成純黑
    for a in range(col):
        for b in range(row):
            if labels[a, b] == 1:
                result[a, b] = 255
            else:
                result[a, b] = 0

    return result


def putSign(img, src='sign.png'):
    sign = cv.imread(src, cv.IMREAD_GRAYSCALE)
    width, height = sign.shape[:2]
    for i in range(width):
        for j in range(height):
            if sign[i, j] < 50:
                img[i, j] = 255
    return img


class Canny(object):
    def __init__(self, lowThreshold=50, highThreshold=150):
        self.lowThreshold = lowThreshold
        self.highThreshold = highThreshold

    def Canny(self, src):
        blur = cv.GaussianBlur(src, (5, 5), 1.5)

        # 梯度向量
        gradientX = cv.Sobel(np.float32(blur), cv.CV_64F, 1, 0, 3)
        gradientY = cv.Sobel(np.float32(blur), cv.CV_64F, 0, 1, 3)
        gradientMagnitude, gradientAngle = cv.cartToPolar(gradientX, gradientY, angleInDegrees=True)

        # cv.imshow("Gradient",gradientMagnitude.astype(np.uint8))
        result = labelComponent(nms(gradientMagnitude, gradientAngle), self.lowThreshold, self.highThreshold).astype(np.uint8)

        result = putSign(result)

        return result
