import cv2

img = cv2.imread('img/5.JPG', 0)

m = 100
k = 40
for i in range(140):
    m -= 1
    for j in range(k):
        if m >= 0:
            img[i, j+m] = 0
        else:
            img[i, j] = 0
    if m < 1:
        k -= 1

cv2.imwrite('img/deadPhoto.jpg', img)
cv2.imshow('output image', img)
cv2.waitKey()