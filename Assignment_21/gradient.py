import cv2
import numpy as np

img = np.full((255, 255), 255, dtype=np.uint8)

for i in range(255):
    for j in range(255):
        img[i, j] = 255 - i
    
cv2.imwrite('img/gradient.jpg', img)
cv2.imshow('output image', img)
cv2.waitKey()