import cv2
import numpy as np

img = cv2.imread('input/lion.png', 0)

final_img = np.zeros(img.shape)
mask = np.array([[0, -1, 0],
                 [-1, 4, -1],
                 [0, -1, 0]])

rows, cols = img.shape
for i in range(1, rows-1):
    for j in range(1, cols-1):
        filterd_area = img[i-1:i+2, j-1:j+2]
        final_img[i, j] = np.sum(filterd_area * mask)

cv2.imwrite('output/lion-output.jpg', final_img)
        
    