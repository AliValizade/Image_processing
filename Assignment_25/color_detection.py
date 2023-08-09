import cv2
import numpy as np

cap = cv2.VideoCapture('input/video1.mp4')

def Convolution(img, mask_dim):
    mask = np.ones((mask_dim, mask_dim)) / np.power(mask_dim, 2)
    # final_img = np.zeros(img.shape)
    # rows, cols = img.shape
    # for i in range(mask_dim//2, rows-(mask_dim//2)):
    #     for j in range(mask_dim//2, cols-(mask_dim//2)):
    #         filtered_area = img[i-int(mask_dim//2):i+1+int(mask_dim//2), j-int(mask_dim//2):j+1+int(mask_dim//2)]  
    #         final_img[i, j] = np.sum(filtered_area * mask)
    final_img = cv2.filter2D(img, -1, mask)
    return final_img

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filtered_frame = Convolution(frame, 25).astype(np.uint8)
    cv2.rectangle(filtered_frame, (130, 250), (300, 450), (255, 255, 0), 2)
    target_area = np.average(filtered_frame[130:300, 250:450])
    if target_area <= 80:
        cv2.putText(filtered_frame, 'BLACK', (30, 80), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 3)
    elif 90 < target_area < 160:
        cv2.putText(filtered_frame, 'GRAY', (30, 80), cv2.FONT_HERSHEY_DUPLEX, 2, (125, 125, 125), 3)
    elif target_area >= 160:
        cv2.putText(filtered_frame, 'WHITE', (30, 80), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 3)
    
    cv2.imshow('Webcam', filtered_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()