import cv2 as cv
import numpy as np
pics = [[0 for i in range(5)] for j in range(4)]
noiseless_pic = [0 for i in range(4)]
final_pic = np.full((2000, 2000), 0, dtype=np.uint8)

for i in range(4):
    for j in range(5):
        pics[i][j] = cv.imread(f"img/hw2/black hole/{i+1}/{j+1}.JPG", 0)
        noiseless_pic[i] += pics[i][j] / 5
    cv.imwrite(f'img/hw2/black hole/{i+1}/{i+1}-without-noise.jpg', noiseless_pic[i])

final_pic[0:1000, 0:1000] = noiseless_pic[0]
final_pic[0:1000, 1000:2000] = noiseless_pic[1]
final_pic[1000:2000, 0:1000] = noiseless_pic[2]
final_pic[1000:2000, 1000:2000] = noiseless_pic[3]

cv.imwrite(f'img/hw2/black hole/Black_hole_without_noise.jpg', final_pic)
final_pic = cv.resize(final_pic, (700, 700))

cv.imshow('final_pics', final_pic)
cv.waitKey()