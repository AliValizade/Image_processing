import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from imutils.perspective import four_point_transform

parser = argparse.ArgumentParser(description="sudoku live detector by Ali Valizade")
parser.add_argument("--filter_siz", type=int, help="size of GaussianBlur mask", default=5)
args = parser.parse_args()

cap = cv.VideoCapture('input/sudoku.mp4') # live video from webcam --> cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gr_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    sudoku_blured = cv.GaussianBlur(gr_frame, (args.filter_siz, args.filter_siz), 3)

    thresh_sudoku = cv.adaptiveThreshold(sudoku_blured, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

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
        print("I cann't find Sudoku!! ")
    else:
        cv.drawContours(frame, [sudoku_contour], -1, (0, 255, 0), 5)

        points = np.array(sudoku_contour, dtype=np.float32)
        pts = points.squeeze()  # convert array(4, 1, 2) to (4, 2)

        result = four_point_transform(frame, pts)
        result = cv.resize(result, (300, 300))
        cv.imshow('webcam', result)
        if cv.waitKey(1) == ord("s"):
            cv.imwrite("output/cropped-sudoku-live.jpg", result)

    if cv.waitKey(1) == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()