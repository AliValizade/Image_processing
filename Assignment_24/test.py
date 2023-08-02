# import cv2
# import numpy as np

# img = np.zeros((500, 500), dtype='uint8')

# points = np.array([[100, 50], [300, 70], [350, 280], [120, 250]])
# cv2.drawContours(img, [points], -1, (255, 255, 255), -1)

# cv2.imshow('output', img)
# cv2.waitKey()


import numpy as np
import cv2
import tensorflow as tf
from functools import partial
import time
from TFLiteFaceDetector import UltraLightFaceDetecion
import sys
from CoordinateAlignment import CoordinateAlignmentModel

def double_size_landmarks(landmarks, center_point):
    diff = landmarks - center_point
    new_landmarks = center_point + 2 * diff
    return new_landmarks

if __name__ == '__main__':
    fd = UltraLightFaceDetecion("weights/RFB-320.tflite", conf_threshold=0.88)
    fa = CoordinateAlignmentModel("weights/coor_2d106.tflite")

    img = cv2.imread('input/20220420_1947030.jpg')
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    rows, cols, _ = img.shape

    mask = np.zeros((rows, cols), dtype='uint8')
    boxes, scores = fd.inference(img)
    for pred in fa.get_landmarks(img, boxes):
        pred_int = np.round(pred).astype(np.int32)
        # Define the landmarks of the eyes and lips as lists of indices
        landmark_left_eye = [35, 36, 33, 37, 39, 42, 40, 41]
        landmark_right_eye = [89, 90, 87, 91, 93, 96, 94, 95]
        landmark_lips = [52, 55, 56, 53, 59, 58, 61, 68, 67, 71, 63, 64]
        # Define the center points of the eyes and lips as tuples of coordinates
        center_point_left_eye = (283, 327)
        center_point_right_eye = (441, 345)
        center_point_lips = (363, 501)
        # Loop over the landmarks and center points
        for landmark, center_point in zip([landmark_left_eye, landmark_right_eye, landmark_lips], [center_point_left_eye,
                                                                                                   center_point_right_eye,
                                                                                                   center_point_lips]):
            # Convert the landmark indices to coordinates
            landmark_coords = np.array([[tuple(pred_int[i]) for i in landmark]])
            # Enlarge the landmark coordinates using the double_size_landmarks function
            enlarged_landmark_coords = double_size_landmarks(landmark_coords, center_point).astype(np.int32)
            # Fill the enlarged landmark with white color using cv2.fillPoly
            cv2.fillPoly(mask, [enlarged_landmark_coords], (255))
            # Find the homography matrix that maps the original landmark to the enlarged one
            H_landmark, _ = cv2.findHomography(landmark_coords, enlarged_landmark_coords)
            # Warp the image using the homography matrix
            warped_landmark = cv2.warpPerspective(img, H_landmark, (cols, rows))
            # Combine the original image and the warped image using the mask
            result = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
            result = cv2.bitwise_or(result, warped_landmark, mask=mask)

        # Blend the original image and the result image using cv2.addWeighted
        final_img = cv2.addWeighted(img, 0.3, result, 0.7, 0)
        # final_img = cv2.add(img, result)

        cv2.imwrite('output/result.jpg', final_img)
        cv2.imshow("result", result)
        cv2.waitKey()
