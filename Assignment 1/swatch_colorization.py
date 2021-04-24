import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys


img = cv2.imread('img1_rgb.jpg')
gray_img = cv2.imread('img1_gray.jpg')
orig_rgb_img = np.copy(img)
orig_gray_img = np.copy(gray_img)
result_img = np.copy(gray_img)
#cv2.imshow('image',img)
clone = img.copy()
refPt1=[]
refPt2=[]
startPt_rgb = []
endPt_rgb = []
startPt_gray = []
endPt_gray = []
cropping = False
#-------function for mouse click event--------------
def click_and_crop(event,x,y,flags,param):
    global refPt1,cropping
    if event==cv2.EVENT_LBUTTONDOWN:
        refPt1 = [(x,y)]
        startPt_rgb.append((x,y))
        cropping = True
    elif event ==cv2.EVENT_LBUTTONUP:
        refPt1.append((x,y))
        endPt_rgb.append((x,y))
        cropping = False
        cv2.rectangle(img,refPt1[0],refPt1[1],(0,255,0),2)
def click2(event,x,y,flags,param):
    global refPt2,cropping
    if event==cv2.EVENT_LBUTTONDOWN:
        refPt2 = [(x,y)]
        startPt_gray.append((x,y))
        cropping = True
    elif event ==cv2.EVENT_LBUTTONUP:
        refPt2.append((x,y))
        endPt_gray.append((x,y))
        cropping = False
        cv2.rectangle(gray_img,refPt2[0],refPt2[1],(0,0,255),2)
        
cv2.namedWindow('image')
cv2.namedWindow('image1')
cv2.setMouseCallback('image',click_and_crop)
cv2.setMouseCallback('image1',click2)
while True:
    cv2.imshow('image',img)
    cv2.imshow('image1',gray_img)
    key = cv2.waitKey(1) & 0xFF
    if key==ord("r"):
        img = clone.copy()
    elif key==ord("c"):
        break
    
#print(refPt1)
#print(refPt2)
#print(orig_rgb_img.shape)
#print(startPt_rgb)
#print(endPt_rgb)
#print(startPt_gray)
#print(endPt_gray)
cv2.destroyAllWindows()
#------swatches are formed so far by startPt and 

#---------making a mask of 5*5 neighborhood--------------------------------
mask = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])

#------convert the color space from rgb to lab---------
rgb_lab = cv2.cvtColor(orig_rgb_img,cv2.COLOR_BGR2Lab)
gray_lab = cv2.cvtColor(orig_gray_img,cv2.COLOR_BGR2Lab)

#----added a padding of (4,4,4,4) top,bottom,left,right pixel of zeros-----------------
padded_orig_img = np.pad(rgb_lab,((4,4),(4,4),(0,0)),'constant')       
padded_gray_img = np.pad(gray_lab,((4,4),(4,4),(0,0)),'constant')

#-------global colormap of gray swatches to used for colorization--------
global_gray_color_map = []


def swatches_color_transfer(rgbX1,rgbY1,rgbX2,rgbY2,grayX1,grayY1,grayX2,grayY2):
    #-- get the sample size of rgb swatch
    no_of_samples = 50
    sample_size1 = np.sqrt(((rgbX2-rgbX1)*(rgbY2-rgbY1))/no_of_samples).astype(int)
    swatch_rgb_color_map = []
    color_map_swatches(padded_orig_img,rgbX1,rgbY1,rgbX2,rgbY2,swatch_rgb_color_map,sample_size1)
    color_transfer_gray_swatches(padded_gray_img,grayX1,grayY1,grayX2,grayY2,swatch_rgb_color_map)
    sample_size_gray = np.sqrt(((grayX2-grayX1)*(grayY2-grayY1))/no_of_samples).astype(int)
    color_map_swatches(padded_gray_img,grayX1,grayY1,grayX2,grayY2,global_gray_color_map,sample_size_gray)

#--------color map for intensity, sd, and colors of the random pixel from rgb samples
def color_map_swatches(image,x,y,m,n,color_map,sample_size):
    i=x
    j=y
    while(i<m-sample_size):
        j=y+2
        while(j<n-sample_size):
            row = np.random.randint(i,i+sample_size)
            col = np.random.randint(j,j+sample_size)
            intensity = (np.mean(np.multiply(image[row-2:row+3,col-2:col+3,0],mask))).astype(int)
            sd = np.std(image[row-2:row+3,col-2:col+3,0])
            color_map.append([intensity,sd,image[row][col][0],image[row][col][1],image[row][col][2]])
            j+=sample_size
        i+=sample_size
#--------color transfer of swatches in the gray image------
def color_transfer_gray_swatches(image,x,y,m,n,color_map):
    i=x
    j=y
    while(i<m):
        j=y
        while(j<n):
            intens = np.mean(np.multiply(image[i-2:i+3,j-2:j+3,0],mask)).astype(int)
            sd1 = np.std(image[i-2:i+3,j-2:j+3,0])
            value = 0.5*intens+0.5*sd1
            index = find(color_map,value)
            image[i][j][1] = color_map[index][3]
            image[i][j][2] = color_map[index][4]
            j+=1
        i+=1

#--------look in the rgb color map to transfer color to gray swatches---------
def find(color_map,val):
    index = -1
    dist = sys.maxsize
    for i,row in enumerate(color_map):
        sds = val-(0.5*row[0]+0.5*row[1])
        if(dist>=np.absolute(sds)):
            dist = np.absolute(sds)
            index = i
    return index
#-------look in the global color gray map to find the neighborhood intensity------
def find1(color_map,val):
    index = -1
    dist = sys.maxsize
    for i,row in enumerate(color_map):
        sds = val-row[0]
        if(dist>=np.absolute(sds)):
            dist = np.absolute(sds)
            index = i
    return index

#---------colorize the gray image--------------------
def colorize_gray_image(image):
    i=4
    j=4
    m = image.shape[0]
    n = image.shape[1]
    while(i<m-5):
        j=4
        while(j<n-5):
            intensity = np.mean(np.multiply(image[i-2:i+3,j-2:j+3,0],mask)).astype(int)
            index = find1(global_gray_color_map,intensity)
            image[i][j][1] = global_gray_color_map[index][3]
            image[i][j][2] = global_gray_color_map[index][4]
            j+=1
        i+=1
#-------step to execute above steps----
for i in range(len(startPt_rgb)):
    swatches_color_transfer(startPt_rgb[i][1],startPt_rgb[i][0],endPt_rgb[i][1],endPt_rgb[i][0],startPt_gray[i][1],startPt_gray[i][0],endPt_gray[i][1],endPt_gray[i][0])
colorize_gray_image(padded_gray_img)
new_img = cv2.cvtColor(padded_gray_img,cv2.COLOR_Lab2BGR)
n1 = new_img.shape[0]
n2 = new_img.shape[1]
result = new_img[4:n1-4,4:n2-4,:]


#cv2.imshow('color',padded_gray_img)
cv2.imshow('color1',result)



plt.subplot(3,2,1),plt.imshow(cv2.cvtColor(orig_rgb_img,cv2.COLOR_BGR2RGB),cmap = 'gray')
plt.title('Original Rgb Image'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,2),plt.imshow(cv2.cvtColor(rgb_lab,cv2.COLOR_BGR2RGB),cmap = 'gray')
plt.title('Original Rgb Lab Image'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,3),plt.imshow(cv2.cvtColor(orig_gray_img,cv2.COLOR_BGR2RGB),cmap = 'gray')
plt.title('Original Gray Image'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,4),plt.imshow(cv2.cvtColor(gray_lab,cv2.COLOR_BGR2RGB),cmap = 'gray')
plt.title('Original Gray Image'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,5),plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB),cmap = 'gray')
plt.title('Colorized Image'), plt.xticks([]), plt.yticks([])
plt.show()

