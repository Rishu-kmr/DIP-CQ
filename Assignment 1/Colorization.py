import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

src=input("Enter Source Image Location: ")         #taking image location
target=input("Enter Target Image Location: ")         #taking image location

#---------------reading images----------------------------------------
img = cv2.imread(src)        #reading the source image
gray_img = cv2.imread(target)  #reading the target image
#---------------converting images to Lab----------------------------
orig_img = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)          # converting source to Lab space
gray_img = cv2.cvtColor(gray_img,cv2.COLOR_BGR2Lab)     # converting target to Lab space
new_img = np.copy(orig_img)                     # making a copy to redraw the image

#---------making a mask of 5*5 neighborhood--------------------------------
mask = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
#------------------some variables------------------------
m = orig_img.shape[0]           # no of rows
n = orig_img.shape[1]           # no of columns
channel = orig_img.shape[2]     # no of channels
no_of_samples = 400             # no of samples user feed
sample_size = np.sqrt((m*n)/no_of_samples).astype(int)

#----added a padding of (4,4,4,4) top,bottom,left,right pixel of zeros-----------------
padded_orig_img = np.pad(orig_img,((4,4),(4,4),(0,0)),'constant')       
padded_gray_img = np.pad(gray_img,((4,4),(4,4),(0,0)),'constant')

color_map=[]            #store the samples color averaged intensity of mask 5*5

#-------making 200 samples and selecting a random pixel from each cell --------------
def make_color_map_for_global_image_matching():
    i=5;
    j=5;
    global color_map
    while(i<m-5):
        j = 5;
        while(j<n-5):
            row = np.random.randint(i,i+sample_size)
            col = np.random.randint(j,j+sample_size)
            intensi = (np.sum(np.multiply(padded_orig_img[row-2:row+3,col-2:col+3,0],mask))/25).astype(int)
            alpha = (np.sum(np.multiply(padded_orig_img[row-2:row+3,col-2:col+3,1],mask))/25).astype(int)
            beta = (np.sum(np.multiply(padded_orig_img[row-2:row+3,col-2:col+3,2],mask))/25).astype(int)
            color_map.append([intensi,padded_orig_img[row][col][0],padded_orig_img[row][col][1],padded_orig_img[row][col][2]])
            #color_map.append([intensi,intensi,alpha,beta])
            j=j+sample_size
        i = i+sample_size

#-----finding the intensity values which resembles more like the input given----------------------
def find(intensity,color_map):
    index = -1;
    dist = sys.maxsize
    for i,rows in enumerate(color_map):
        if(dist>=np.absolute(intensity-rows[0])):
            dist = np.absolute(intensity-rows[0])
            index = i;
    return index
def form_colorization_of_gray_image_by_global_coloring():
    r_ind = 5;
    c_ind = 5;
    global new_img
    while(r_ind<padded_gray_img.shape[0]-5):
        c_ind = 5;
        while(c_ind<padded_gray_img.shape[1]-5):
            intensity = padded_gray_img[r_ind][c_ind][0]
            neighborhood_intensity = (np.sum(np.multiply(padded_gray_img[r_ind-2:r_ind+3,c_ind-2:c_ind+3,0],mask))/25).astype(int)
            index = find(neighborhood_intensity,color_map)
            new_img[r_ind-5][c_ind-5][0] = intensity
            new_img[r_ind-5][c_ind-5][1] = color_map[index][2]
            new_img[r_ind-5][c_ind-5][2] = color_map[index][3]
            c_ind = c_ind+1
        r_ind = r_ind+1
def display_img():
    global new_img
    global gray_img
    new_img = cv2.cvtColor(new_img,cv2.COLOR_Lab2BGR)
    gray_lab_img = np.copy(gray_img)
    gray_img = cv2.cvtColor(gray_img,cv2.COLOR_Lab2BGR)
    plt.subplot(3,2,1),plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(3,2,2),plt.imshow(cv2.cvtColor(orig_img,cv2.COLOR_BGR2RGB),cmap = 'gray')
    plt.title('Original Lab_Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(3,2,3),plt.imshow(cv2.cvtColor(gray_img,cv2.COLOR_BGR2RGB),cmap = 'gray')
    plt.title('Gray Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(3,2,4),plt.imshow(cv2.cvtColor(gray_lab_img,cv2.COLOR_BGR2RGB),cmap = 'gray')
    plt.title('Gray Lab Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(3,2,5),plt.imshow(cv2.cvtColor(new_img,cv2.COLOR_BGR2RGB),cmap = 'gray')
    plt.title('Colorized Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    

def run_global_img_matching():
    make_color_map_for_global_image_matching()
    form_colorization_of_gray_image_by_global_coloring()
    display_img()
run_global_img_matching()
#-----------------end of global_matching---------------------------------------------------------


#------by swatches--------------------------------------

    
