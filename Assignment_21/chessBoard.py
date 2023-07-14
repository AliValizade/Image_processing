import cv2
import numpy as np

img = np.full((800, 800), 255, dtype=np.uint8)

for i in range(800):
    for j in range(800):
        if (i // 100) % 2 == (j // 100) % 2:
            img[i, j] = 0
        else:
            img[i, j] = 255

cv2.imwrite('img/chessBoard.jpg', img)
cv2.imshow('output image', img)
cv2.waitKey()