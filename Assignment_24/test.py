import cv2
import numpy as np

img = np.zeros((500, 500), dtype='uint8')

points = np.array([[100, 50], [300, 70], [350, 280], [120, 250]])
cv2.drawContours(img, [points], -1, (255, 255, 255), -1)

cv2.imshow('output', img)
cv2.waitKey()


