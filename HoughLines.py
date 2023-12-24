import numpy as np
import math
import cv2 as cv


class HoughLines(object):
    def __init__(self, thetaMin=-90, thetaMax=90, threshold=10):
        self.rhoMax = None
        self.rhos = None
        self.thetaMin = thetaMin
        self.thetaMax = thetaMax
        self.threshold = threshold
        self.thetas = np.deg2rad(range(thetaMin, thetaMax, 1))

    def vote(self, image):
        y, x = image.shape[:2]

        self.rhoMax = int(math.hypot(x, y))
        numRhos = self.rhoMax * 2 + 1
        self.rhos = np.linspace(-self.rhoMax, self.rhoMax, numRhos)
        # print(self.rhoMax)
        numThetas = len(self.thetas)
        accumulators = np.zeros((numRhos, numThetas), dtype=np.uint8)

        for y in range(y):
            for x in range(x):
                if image[y, x] > self.threshold:
                    for i in range(numThetas):
                        theta = self.thetas[i]
                        # added for a positive index
                        rho = round(x * math.cos(theta) + y * math.sin(theta)) + self.rhoMax
                        accumulators[rho, i] += 1
        return accumulators
