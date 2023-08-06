import numpy as np
import cv2
import tensorflow as tf
from functools import partial
from TFLiteFaceDetector import UltraLightFaceDetecion
from CoordinateAlignment import CoordinateAlignmentModel

def zoom_effect(img, landmarks):
    x, y, w, h = cv2.boundingRect(landmarks)
    rows, cols, _ = img.shape
    mask = np.zeros((rows, cols, 3), dtype='uint8')
    cv2.drawContours(mask, [landmarks], -1, (255, 255, 255), -1)
    img_2x = cv2.resize(img, None, fx=2, fy=2)
    mask_2x = cv2.resize(mask, None, fx=2, fy=2)

    # Normalization (0, 1)
    img_2x = img_2x / 255
    mask_2x = mask_2x / 255

    landmarks_target = img[int(y - (h / 2)):int(y + h + (h / 2)), int(x - (w / 2)):int(x + w + (w / 2))]
    landmarks_target = landmarks_target / 255

    fg = cv2.multiply(mask_2x, img_2x)
    bg = cv2.multiply(landmarks_target, 1 - mask_2x[y * 2:(y + h) * 2, x * 2:(x + w) * 2])
    res = cv2.add(bg, fg[y * 2:(y + h) * 2, x * 2:(x + w) * 2])

    # convert img target areas from (0, 1) to (0, 255)
    img[int(y - (h / 2)):int(y + h + (h / 2)), int(x - (w / 2)):int(x + w + (w / 2))] = res * 255
    return img


if __name__ == '__main__':
    fd = UltraLightFaceDetecion("weights/RFB-320.tflite", conf_threshold=0.88)
    fa = CoordinateAlignmentModel("weights/coor_2d106.tflite")

    img = cv2.imread('input/ali.jpg')
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    boxes, scores = fd.inference(img)
    for pred in fa.get_landmarks(img, boxes):
        pred_int = np.round(pred).astype(np.int32)
        landmark_left_eye = np.array([[tuple(pred_int[i]) for i in [35, 36, 33, 37, 39, 42, 40, 41]]])
        landmark_right_eye = np.array([[tuple(pred_int[i]) for i in [89, 90, 87, 91, 93, 96, 94, 95]]])
        landmark_lips = np.array([[tuple(pred_int[i]) for i in [52, 55, 56, 53, 59, 58, 61, 68, 67, 71, 63, 64]]])
        # print(landmark_left_eye)

        # Find the homography matrix that maps the original landmarks to the enlarged ones
        # H_L_eye, _ = cv2.findHomography(landmark_left_eye, enlarged_landmark_left_eye)
        # H_R_eye, _ = cv2.findHomography(landmark_right_eye, enlarged_landmark_right_eye)
        # H_lips, _ = cv2.findHomography(landmark_lips, enlarged_landmark_lips)
        # Warp the image using the homography matrix
        # warped_L = cv2.warpPerspective(img, H_L_eye, (cols, rows))
        # warped_R = cv2.warpPerspective(img, H_R_eye, (cols, rows))
        # warped_lips = cv2.warpPerspective(img, H_lips, (cols, rows))

        result = zoom_effect(img, landmark_left_eye)
        result = zoom_effect(img, landmark_right_eye)
        result = zoom_effect(img, landmark_lips)
        # print(landmark_left_eye)
        # for index, p in enumerate(np.round(pred).astype(np.int32)):
        #     print(p, index)
        #     cv2.circle(img, tuple(p), 1, (125, 255, 125), 1, cv2.LINE_AA)
        #     cv2.putText(img, str(index), p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # result = result // 255
    # result = img * result
    # final_img = cv2.add(img, result)

    cv2.imwrite('output/result.jpg', result)
    cv2.imshow("result", result)
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
