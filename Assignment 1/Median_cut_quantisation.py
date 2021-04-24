import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

img = cv2.imread('test.jpeg')
orig_img = np.copy(img)
img_quan = np.copy(img)
n = 5; #define in the power of 2 for buckets that is (#buckets = 2^n = k)
no_rows = img.shape[0]
no_cols = img.shape[1]
no_channels = img.shape[2]
color_map = []

#--------------UNIFORM QUANTISATION---------------------------
new_img = np.copy(img)
r_region_mapping = [[],[],[],[],[],[],[],[]]
g_region_mapping = [[],[],[],[],[],[],[],[]]
b_region_mapping = [[],[],[],[],[],[],[],[]]
r_representative_color_per_region = [0,0,0,0,0,0,0,0]
g_representative_color_per_region = [0,0,0,0,0,0,0,0]
b_representative_color_per_region = [0,0,0,0,0,0,0,0]
def get_region_index(color):
    eight_regions = [[0,31],[32,63],[64,95],[96,127],[128,159],[160,191],[192,223],[224,255]]
    for index,region in enumerate(eight_regions):
        if(color>=region[0] and color<=region[1]):
            return index
def assign_given_image_colors_to_ranges():
    for rows in img:
        for pixel in rows:
            red = pixel[0]
            green = pixel[1]
            blue = pixel[2]
            r_region_mapping[get_region_index(red)].append(red)
            g_region_mapping[get_region_index(red)].append(green)
            b_region_mapping[get_region_index(red)].append(blue)
def uniform_quantisation(img):
    assign_given_image_colors_to_ranges()
    assign_calc_final_color()
    for r_index,rows in enumerate(img):
        for c_index,pixel in enumerate(rows):
            red = pixel[0]
            green = pixel[1]
            blue = pixel[2]
            new_img[r_index][c_index][0] = r_representative_color_per_region[get_region_index(red)]
            new_img[r_index][c_index][1] = g_representative_color_per_region[get_region_index(green)]
            new_img[r_index][c_index][2] = b_representative_color_per_region[get_region_index(blue)]
def assign_calc_final_color():
    for i in range(8):
        r_representative_color_per_region[i] = np.mean(r_region_mapping[i]).astype(int)
        g_representative_color_per_region[i] = np.mean(g_region_mapping[i]).astype(int)
        b_representative_color_per_region[i] = np.mean(b_region_mapping[i]).astype(int)
uniform_quantisation(img)

#--------------END OF UNIFORM QUANTISATION-------------------------------
def user_input(img,k):
    return;
def distance(a,b,c,d,e,f):
    return (np.square(a-d)+np.square(b-e)+np.square(c-f))
def inRange(i,j):
    return i>=0 and i<no_rows and j>=0 and j<no_cols;
def form_color_map(img,img_arr):
    r_avg = np.mean(img_arr[:,0])
    g_avg = np.mean(img_arr[:,1])
    b_avg = np.mean(img_arr[:,2])
    global color_map
    color_map.append([r_avg,g_avg,b_avg])
def popularity_algorithm(img):
    hist_blue = cv2.calcHist([img],[0],None,[256],[0,256])
    hist_green = cv2.calcHist([img],[1],None,[256],[0,256])
    hist_red = cv2.calcHist([img],[2],None,[256],[0,256])
    hist_b = np.sort(np.array(hist_blue))[::-1]
    hist_g = np.sort(np.array(hist_green))[::-1]
    hist_r = np.sort(np.array(hist_red))[::-1]
    color_map_popularity = cv2.merge((hist_b,hist_g,hist_r))
    np.array(color_map_popularity)
    print(color_map_popularity.shape)
def find(r,g,b,color_map):
    index = 0;
    dist = sys.maxsize
    for i,rows in enumerate(color_map):
        r1 = rows[0]
        g1 = rows[1]
        b1 = rows[2]
        if(dist>=distance(r,g,b,r1,g1,b1)):
            dist = distance(r,g,b,r1,g1,b1)
            index = i;
    return index;
def split_box(img,img_arr,k):
    if(len(img_arr)==0):
        return
    if(k==0):
        form_color_map(img,img_arr)
        return;
    r_range = np.max(img_arr[:,0])-np.min(img_arr[:,0])
    g_range = np.max(img_arr[:,1])-np.min(img_arr[:,1])
    b_range = np.max(img_arr[:,2])-np.min(img_arr[:,2])
    color_with_max_range = -1;
    
    if(r_range>=g_range and r_range>=b_range):
        color_with_max_range = 0;
    elif(g_range>=r_range and g_range>=b_range):
        color_with_max_range = 1;
    else:
        color_with_max_range = 2;
    img_arr = img_arr[img_arr[:,color_with_max_range].argsort()]
    median = int((len(img_arr)+1)/2)
    split_box(img,img_arr[0:median],k-1)
    split_box(img,img_arr[median:],k-1)
def redraw_image():
    full_arr_box = []
    for r_index,rows in enumerate(img):
        for c_index,color in enumerate(rows):
            full_arr_box.append([color[0],color[1],color[2],r_index,c_index])
    full_arr_box = np.array(full_arr_box)
    split_box(img,full_arr_box,n)           #calling the function to form k colors
    global color_map
    color_map = np.array(color_map)
    global img_quan
    img_quan = np.copy(img)
    for r_index,rows in enumerate(img):
        for c_index,pixel in enumerate(rows):
            red = pixel[0]
            green = pixel[1]
            blue = pixel[2]
            index = find(red,green,blue,color_map)
            img_quan[r_index][c_index] = [color_map[index,0],color_map[index,1],color_map[index,2]]
            e1 = red-color_map[index,0];
            e2 = green-color_map[index,1];
            e3 = blue-color_map[index,2];
            #e = [(red-color_map[index,0]),(green-color_map[index,1]),(blue-color_map[index,2])]
            if(inRange(r_index,c_index+1)):
                img[r_index][c_index+1][0] = img[r_index][c_index+1][0]+e1*(3/8);
                img[r_index][c_index+1][1] = img[r_index][c_index+1][1]+e2*(3/8);
                img[r_index][c_index+1][2] = img[r_index][c_index+1][2]+e3*(1/4);
            if(inRange(r_index+1,c_index)):
                img[r_index+1][c_index][0] = img[r_index+1][c_index][0]+e1*(3/8);
                img[r_index+1][c_index][1] = img[r_index+1][c_index][1]+e2*(3/8);
                img[r_index+1][c_index][2] = img[r_index+1][c_index][2]+e3*(1/4);
            if(inRange(r_index+1,c_index+1)):
                img[r_index+1][c_index+1][0] = img[r_index+1][c_index+1][0]+e1*(3/8);
                img[r_index+1][c_index+1][1] = img[r_index+1][c_index+1][1]+e2*(3/8);
                img[r_index+1][c_index+1][2] = img[r_index+1][c_index+1][2]+e3*(1/4);
    display_img()
def display_img():
    plt.subplot(2,2,1),plt.imshow(cv2.cvtColor(orig_img,cv2.COLOR_BGR2RGB),cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(cv2.cvtColor(new_img,cv2.COLOR_BGR2RGB),cmap = 'gray')
    plt.title('Uniform Quantized Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),cmap = 'gray')
    plt.title('Dithered Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(cv2.cvtColor(img_quan,cv2.COLOR_BGR2RGB),cmap = 'gray')
    plt.title('Median Quantized Image'), plt.xticks([]), plt.yticks([])
    plt.show()


redraw_image()
