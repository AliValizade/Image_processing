import cv2

img1 = cv2.imread('img/1.jpg', 0)
img2 = cv2.imread('img/2.jpg', 0)

img1 = 255 - img1
img2 = 255 - img2

cv2.imwrite('img/real_1.jpg', img1)
cv2.imshow('output image', img1)
cv2.waitKey()

cv2.imwrite('img/real_2.jpg', img2)
cv2.imshow('output image', img2)
cv2.waitKey()