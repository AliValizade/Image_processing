import cv2
import numpy as np

img = cv2.imread('input/building.tif', 0)

final_img1 = np.zeros(img.shape)
final_img2 = np.zeros(img.shape)

mask1 = np.array([[-1, 0, 1],
                 [-1, 0, 1],
                 [-1, 0, 1]])

mask2 = np.array([[-1, -1, -1],
                  [0, 0 ,0],
                  [1, 1, 1]])

rows, cols = img.shape
for i in range(1, rows-1):
    for j in range(1, cols-1):
        filtered_area = img[i-1:i+2, j-1:j+2]
        final_img1[i, j] = np.sum(filtered_area * mask1)
        final_img2[i, j] = np.sum(filtered_area * mask2)

cv2.imwrite('output/building-1.jpg', final_img1)
cv2.imwrite('output/building-2.jpg', final_img2)