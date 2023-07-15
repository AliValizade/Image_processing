import cv2 as cv
import numpy as np

images = [0 for i in range(15)]
final_image = np.full((240, 320), 0, dtype=np.uint8)
for i in range(15):
    images[i] = cv.imread(f'img/hw2/highway/h{i}.JPG', 0)
    final_image += images[i] // 15

cv.imwrite('img/hw2/highway/final_highway.jpg', final_image)
cv.imshow('final image', final_image)
cv.waitKey()