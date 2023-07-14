import cv2
import numpy as np

img = np.full((400, 400), 255, dtype=np.uint8)

m = 250
for i in range(200):
    m -= 1
    for j in range(40):
        img[i+100, j+m] = 0
        img[i+100, j+250] = 0
    
for i in range(20):
    for j in range(140):
        img[i+200, j+150] = 0

cv2.imwrite('img/name.jpg', img)
cv2.imshow('output image', img)
cv2.waitKey()
