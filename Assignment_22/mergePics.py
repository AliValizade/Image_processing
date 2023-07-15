import cv2 as cv
import numpy as np

ali = cv.imread('img/hw2/Ali.jpg', 0)
elon = cv.imread('img/hw2/Elon_Musk.jpg', 0)

mergeFace_1 = ali // 2 + elon // 4
mergeFace_2 = ali // 4 + elon // 2

picFrame = np.full((400, 1600), 0, dtype=np.uint8)

picFrame[0:400, 0:400] = ali
picFrame[0:400, 400:800] = mergeFace_1
picFrame[0:400, 800:1200] = mergeFace_2
picFrame[0:400, 1200:1600] = elon

cv.imwrite('img/hw2/picFrame_of_mergeFaces.jpg', picFrame)
cv.imshow('merge faces', picFrame)
cv.waitKey()