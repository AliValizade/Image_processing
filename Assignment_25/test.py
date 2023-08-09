import cv2
import numpy as np

img = cv2.imread('input/253629_6.png.webp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

result = np.zeros(img.shape)
filter_size = 3     # (3*3)
filter_len = 9
rows, cols = img.shape
for i in range(filter_size//2, rows-(filter_size//2)):
    for j in range(filter_size//2, cols-(filter_size//2)):
        small_img = img[i-(filter_size//2):i+(filter_size//2)+1, j-(filter_size//2):j+(filter_size//2)+1]
        small_img_1d = small_img.reshape(filter_len)
        small_img_1d_sorted = np.sort(small_img_1d)
        result[i, j] = small_img_1d_sorted[filter_len//2]

cv2.imwrite('output/result-best3.jpg', result)    
# cv2.imwrite('output/result-best5.jpg', result)    