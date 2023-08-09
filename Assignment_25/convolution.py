import cv2
import numpy as np

img = cv2.imread('input/Elon_Musk.jpg', 0)

def Convolution(img, mask_dim):
    final_img = np.zeros(img.shape)
    mask = np.ones((mask_dim, mask_dim)) / np.power(mask_dim, 2)
    rows, cols = img.shape
    for i in range(mask_dim//2, rows-(mask_dim//2)):
        for j in range(mask_dim//2, cols-(mask_dim//2)):
            filtered_area = img[i-int(mask_dim//2):i+1+int(mask_dim//2), j-int(mask_dim//2):j+1+int(mask_dim//2)]  
            final_img[i, j] = np.sum(filtered_area * mask)
    return final_img

mask_dimension = int(input('Enter the dimensions of the convolution filter ((n,n) as n): '))
cv2.imwrite(f'output/image{mask_dimension}.jpg', Convolution(img, mask_dimension))      