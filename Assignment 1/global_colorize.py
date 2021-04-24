import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

img = cv2.imread('test_rgb.png')        #reading an image
gray_img = cv2.imread('test_gray.png')
# converting color and gray images to Lab image
img_lab = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
gray_img_lab = cv2.cvtColor(gray_img,cv2.COLOR_BGR2Lab)
new_img = np.copy(gray_img_lab)
final_img = cv2.cvtColor(gray_img_lab,cv2.COLOR_Lab2BGR)




plt.subplot(3,2,1),plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,2),plt.imshow(cv2.cvtColor(img_lab,cv2.COLOR_BGR2RGB),cmap = 'gray')
plt.title('Original Luv_Image'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,3),plt.imshow(cv2.cvtColor(gray_img,cv2.COLOR_BGR2RGB),cmap = 'gray')
plt.title('Gray Image'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,4),plt.imshow(cv2.cvtColor(gray_img_lab,cv2.COLOR_BGR2RGB),cmap = 'gray')
plt.title('Gray Luv_Image'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,5),plt.imshow(cv2.cvtColor(final_img,cv2.COLOR_BGR2RGB),cmap = 'gray')
plt.title('Colorized Image'), plt.xticks([]), plt.yticks([])
plt.show()
