import cv2 as cv
import numpy as np

origin_board = cv.imread('img/hw2/board - origin.bmp', 0)
test_board = cv.imread('img/hw2/board - test.bmp', 0)

origin_board = cv.rotate(origin_board, cv.ROTATE_90_CLOCKWISE)
test_board = cv.rotate(test_board, cv.ROTATE_90_COUNTERCLOCKWISE)
test_board = cv.flip(test_board, 180)

difference = cv.subtract(origin_board, test_board)

cv.imwrite('img/hw2/difference_of_boards.BMP', difference)
cv.imshow('original board', origin_board)
cv.imshow('test board', test_board)
cv.imshow('difference of boards', difference)
cv.waitKey()
