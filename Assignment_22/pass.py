import cv2 as cv

img1 = cv.imread('img/hw2/a.tif', 0)
img2 = cv.imread('img/hw2/b.tif', 0)

password = img2 - img1

cv.imshow('image a', img1)
cv.imshow('image b', img2)
cv.imwrite('img/hw2/sub_b-a.jpg', password)
cv.imshow('substract of a & b', password)
cv.waitKey()