import numpy as np
import cv2

img = cv2.imread('lena.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img2 = np.copy(img)
print(img.shape)
std = np.std(img)
mean = np.mean(img)
img = np.pad(img,((4,4),(4,4)),'constant')
img2 = np.pad(img,((4,4),(4,4)),'constant')
m,n = img.shape
i = 5;
j = 5;
while(i<m):
    j=5;
    while(j<n):
        img2[i][j] = np.std(img[i-3:i+3,j-3:j+3])
        j+=1
    i+=1
print(std)
cv2.imshow('image',img2)
