import cv2
import numpy as np

emoji = cv2.imread('img/emoji_1.png')
eyes_sticker = cv2.imread('img/eye.png')
lips_sticker = cv2.imread('img/lips.png')

face_cascade = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')
eyes_detector = cv2.CascadeClassifier('files/haarcascade_eye.xml')
lips_detector = cv2.CascadeClassifier('files/haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

# Define a function to overlay a sticker on an image
def apply_sticker(img, sticker, x, y, w, h):
    sticker = cv2.resize(sticker, (w, h))
    sticker_gray = cv2.cvtColor(sticker, cv2.COLOR_BGR2GRAY)
    # Create a mask for the sticker
    _, mask = cv2.threshold(sticker_gray, 1, 255, cv2.THRESH_BINARY)
    # Invert the mask
    mask_inv = cv2.bitwise_not(mask)
    # Extract the region of interest from the image (roi)
    roi = img[y:y+h, x:x+w]
    # Remove the background of the sticker from the roi
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    # Add the sticker to the roi
    roi_fg = cv2.bitwise_and(sticker, sticker, mask=mask)
    # Combine the roi_bg and roi_fg
    final_sticker = cv2.add(roi_bg, roi_fg)
    # Replace the roi in the image with the final_sticker
    img[y:y+h, x:x+w] = final_sticker

choice = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    gray_fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_fr, 1.3, 5)
    for (x, y, w, h) in faces:
        if choice == 1:
            # Overlay emoji on the face
            apply_sticker(frame, emoji, x, y, w, h)
        elif choice == 2:
            gray_fr_roi = gray_fr[y:y+h, x:x+w]
            fr_roi = frame[y:y+h, x:x+w]
            eyes = eyes_detector.detectMultiScale(gray_fr_roi, 1.1, 5)
            for (x, y, w, h) in eyes:
                apply_sticker(fr_roi, eyes_sticker, x, y, w, h)
        elif choice == 3:
            gray_fr_roi = gray_fr[y:y+h, x:x+w]
            fr_roi = frame[y:y+h, x:x+w]
            lips = lips_detector.detectMultiScale(gray_fr_roi, 1.8, 8)
            for (x, y, w, h) in lips:
                apply_sticker(fr_roi, lips_sticker, x, y, w, h)
        elif choice == 4:
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            # for i in range(4):
            #     for j in range(4):
            #         if (i + j) % 2 == 0:
            #             color = (50, 50, 50) 
            #         else:
            #             color = (150, 150, 150) 
            #         x1 = x + i * (w // 4)
            #         y1 = y + j * (h // 4)
            #         x2 = x1 + (w // 4)
            #         y2 = y1 + (h // 4)
            #         cv2.addWeighted(frame, 0.7, cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1), 0.3, 0)
            temp=cv2.resize(frame[y:y+h,x:x+w],(16,16),interpolation=cv2.INTER_LINEAR)
            frame[y:y+h,x:x+w]=cv2.resize(temp,(w,h),interpolation=cv2.INTER_NEAREST)  

        elif choice == 5:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    cv2.imshow('Webcam', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break
    elif key == ord('1'):
        choice = 1
    elif key == ord('2'):
        choice = 2
    elif key == ord('3'):
        choice = 3
    elif key == ord('4'):
        choice = 4
    elif key == ord('5'):
        choice = 5

# cap.release()
# cv2.destroyAllWindows()
