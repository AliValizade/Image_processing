import cv2 as cv
from random import randint

img = cv.imread('img/hw2/difference_of_boards.BMP', 0)
img = cv.resize(img, (800, 600))

height, width = img.shape
for i in range(height):
    for j in range(width):
        if randint(1, height) < 2:
            img[i, j] = randint(0, 255)

cv.imwrite('img/hw2/noise.jpg', img)
cv.imshow('output image', img)
cv.waitKey()

