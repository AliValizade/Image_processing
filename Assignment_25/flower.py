import cv2
import numpy as np

img = cv2.imread('input/flower_input.jpg')
gr_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mask = np.ones((25, 25)) / 625
final_img = np.zeros(gr_img.shape)

rows, cols = gr_img.shape
for i in range(12, rows-12):
    for j in range(12, cols-12):
        if gr_img[i, j] < 180:
            filtered_area = gr_img[i-12:i+13, j-12:j+13]
            final_img[i, j] = np.sum(filtered_area * mask)
        else:
            final_img[i, j] = gr_img[i, j]

cv2.imwrite('output/folower-output.jpg', final_img)
                
    