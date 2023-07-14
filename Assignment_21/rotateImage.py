import cv2

img = cv2.imread('img/3.jpg', 0)
img = cv2.resize(img, (900, 495))

height , width = img.shape

rotated_img = cv2.rotate(img, cv2.ROTATE_180)

cv2.imwrite('img/rotated_3.jpg', rotated_img)
cv2.imshow('output image', rotated_img)
cv2.waitKey()