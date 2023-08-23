import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from imutils.perspective import four_point_transform

parser = argparse.ArgumentParser(description="sudoku live detector by Ali Valizade")
parser.add_argument('--input', type=str, help='path of your input image', default='input/sudoku.jpg')
parser.add_argument("--filter_size", type=int, help="size of GaussianBlur mask", default=7)
args = parser.parse_args()

sudoku = cv.imread(args.input)
gray_sudoku = cv.cvtColor(sudoku, cv.COLOR_BGR2GRAY)

sudoku_blured = cv.GaussianBlur(gray_sudoku, (args.filter_size, args.filter_size), 3)

thresh_sudoku = cv.adaptiveThreshold(sudoku_blured, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

thresh_sudoku = cv.bitwise_not(thresh_sudoku)

contours = cv.findContours(thresh_sudoku, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = contours[0]

contours = sorted(contours, key=cv.contourArea, reverse=True)

sudoku_contour = None
for contour in contours:
    epsilon = 0.02 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)
    if len(approx) == 4:
        sudoku_contour = approx
        break
if sudoku_contour is None:
    print('Sorry, Sudoku not found!')

result = cv.drawContours(sudoku, [sudoku_contour], -1, (0, 255, 0), 5)
# plt.imshow(result)

points = np.array(sudoku_contour, dtype=np.float32)
pts = points.squeeze() # convert array(4, 1, 2) to (4, 2)

cropped_sudoku = four_point_transform(sudoku, pts)
cv.imwrite('output/cropped-sudoku.jpg', cropped_sudoku)
cv.imshow('sudo', cropped_sudoku)
cv.waitKey()