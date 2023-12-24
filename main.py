import cv2
import numpy as np
from matplotlib import pyplot as plt

import Canny
import HoughLines

img = cv2.imread('street2.jpg', cv2.IMREAD_GRAYSCALE)
# cv2.imshow('img', img)
# canny edge detection
canny = Canny.Canny()
edges = canny.Canny(img)

cv2.imshow('Canny', edges)
cv2.imwrite('str2canny.jpg', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# hough transform
hough = HoughLines.HoughLines()
vote = hough.vote(edges)

fig = plt.figure(figsize=(10, 10))
plt.imshow(vote, cmap='inferno')
plt.ylabel('theta')
plt.xlabel('rho')
plt.margins(x=0.1)
# plt.show()
plt.savefig('str2vote.jpg')


