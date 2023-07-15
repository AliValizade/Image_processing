import cv2 as cv

myPic = cv.imread('img/hw2/Ali.jpg', 0)
inverted_pic = 255 - myPic
blurred_pic = cv.GaussianBlur(inverted_pic, (21, 21), 0)
inverted_blurred_pic = 255 - blurred_pic
sketch_pic = myPic / inverted_blurred_pic
sketch_pic *= 255

cv.imwrite('img/hw2/paint_of_myPic.jpg', sketch_pic)
cv.imshow('output', sketch_pic)
cv.waitKey()