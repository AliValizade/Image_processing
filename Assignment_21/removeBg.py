import cv2

img = cv2.imread('img/4.jpg', 0)
img = cv2.resize(img, (1200, 750))

height, width = img.shape
for i in range(height):
    for j in range(width):
        if img[i, j] < 126:
            img[i , j] = 0

cv2.imwrite('img/removeBg_4.jpg', img)
cv2.imshow('output image', img)
cv2.waitKey()