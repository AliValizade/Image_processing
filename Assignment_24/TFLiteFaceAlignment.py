import numpy as np
import cv2
import tensorflow as tf
from functools import partial
import time
from TFLiteFaceDetector import UltraLightFaceDetecion
import sys
from CoordinateAlignment import CoordinateAlignmentModel

def double_size_landmarks(landmarks, center_point):
    new_landmarks = []
    for landmark in landmarks:
        diff = np.array(landmark) - np.array(center_point)
        new_landmark = center_point + 2 * diff
        new_landmarks.append(new_landmark)
    return new_landmarks

if __name__ == '__main__':
    fd = UltraLightFaceDetecion("weights/RFB-320.tflite", conf_threshold=0.88)
    fa = CoordinateAlignmentModel("weights/coor_2d106.tflite")

    img = cv2.imread('input/20220420_1947030.jpg')
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    rows = img.shape[0]
    cols = img.shape[1]
    mask = np.zeros((rows, cols, 3), dtype='uint8')

    boxes, scores = fd.inference(img)
    for pred in fa.get_landmarks(img, boxes):
        pred_int = np.round(pred).astype(np.int32)
        landmark_left_eye = np.array([[tuple(pred_int[i]) for i in [35, 36, 33, 37, 39, 42, 40, 41]]])
        landmark_right_eye = np.array([[tuple(pred_int[i]) for i in [89, 90, 87, 91, 93, 96, 94, 95]]])
        landmark_lips = np.array([[tuple(pred_int[i]) for i in [52, 55, 56, 53, 59, 58, 61, 68, 67, 71, 63, 64]]])
        # print(landmark_left_eye)
        landmark_left_eye = double_size_landmarks(landmark_left_eye, (283, 327))
        landmark_right_eye = double_size_landmarks(landmark_right_eye, (441, 345))
        landmark_lips = double_size_landmarks(landmark_lips, (363, 501))

        cv2.drawContours(mask, landmark_left_eye, -1, (255, 255, 255), -1)
        cv2.drawContours(mask, landmark_right_eye, -1, (255, 255, 255), -1)
        cv2.drawContours(mask, landmark_lips, -1, (255, 255, 255), -1)
        # print(landmark_left_eye)
        # for index, p in enumerate(np.round(pred).astype(np.int32)):
        #     print(p, index)
        #     cv2.circle(img, tuple(p), 1, (125, 255, 125), 1, cv2.LINE_AA)
        #     cv2.putText(img, str(index), p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    mask = mask // 255
    result = img * mask
    final_img = cv2.add(img, result)

    cv2.imwrite('output/result.jpg', final_img)
    cv2.imshow("result", final_img)
    cv2.waitKey()
        

# if __name__ == '__main__':
#     fd = UltraLightFaceDetecion("weights/RFB-320.tflite", conf_threshold=0.88)
#     fa = CoordinateAlignmentModel("weights/coor_2d106.tflite")

#     cap = cv2.VideoCapture(sys.argv[1])
#     color = (125, 255, 125)

#     while True:
#         ret, frame = cap.read()

#         if not ret:
#             break

#         start_time = time.perf_counter()

#         boxes, scores = fd.inference(frame)

#         for pred in fa.get_landmarks(frame, boxes):
#             for p in np.round(pred).astype(np.int):
#                 cv2.circle(frame, tuple(p), 1, color, 1, cv2.LINE_AA)

#         print(time.perf_counter() - start_time)

#         cv2.imshow("result", frame)
#         if cv2.waitKey(0) == ord('q'):
#             break
