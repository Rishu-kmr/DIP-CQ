import cv2
import numpy as np
import matplotlib.pyplot as plt



img = cv2.imread('lena.png')
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
cv2.imshow('image',new_img)
