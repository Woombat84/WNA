#!/usr/bin/python3
# This is just a rouhg layout

import cv2
import time

import numpy as np
#from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import train_test_split
import pandas as pd


import matplotlib.pyplot as plt



def Skeletonizer(img):

    if img.shape[2] != 1:
       img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)

    ret,img = cv2.threshold(img,127,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    
    while( not done):
        
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
    

    
    return skel 


# image aqustion and pre-prosseccing is now done it is time for 
# segmentation

# bluring
# edge detection
# thresholding on color hsv

# this function segment the input image
# depened on what representation the image
# is in the output is a binary image. 
def segmention(img,color):
    # 0 color
    # 1 HSV
    # 2 gray
    # erode
    wn_color = "color"
    wn_hsv ="hsv"
    wn_gray = "gray"

    col_pre_intv = 10
    gray_pre_intv = 10
    hsv_pre_intv = 10

    col_pre_edges = 0.1
    gray_pre_edges = 0.1
    hsv_pre_edges = 0.1

    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    
    if color == 0:
        #cv2.namedWindow(wn_color, cv2.WINDOW_NORMAL)
        blur_img = cv2.GaussianBlur(img, (9, 9), 0)
        col_retval, thres_img = cv2.threshold(blur_img, 255, cv2.THRESH_BINARY, cv2.THRESH_TRUNC)
        canny_img = cv2.Canny(thres_img,col_retval-col_pre_intv,col_retval+col_pre_intv)
        seg_img = cv2.erode(canny_img,element)
        #cv2.imshow(wn_color,canny_img)
    if color == 1:
        #cv2.namedWindow(wn_hsv, cv2.WINDOW_NORMAL)
        blur_img = cv2.GaussianBlur(img, (9, 9), 0)
        hsv_retval, thres_img = cv2.threshold(blur_img, 125, cv2.THRESH_BINARY, cv2.THRESH_TRUNC)
        canny_img = cv2.Canny(blur_img,40,100)
        seg_img = cv2.erode(canny_img,element)
        #cv2.imshow(wn_hsv,canny_img)
    if color == 2:
        #cv2.namedWindow(wn_gray, cv2.WINDOW_NORMAL)
        blur_img = cv2.GaussianBlur(img, (3, 3), 0)
        retval, thres_img = cv2.threshold(blur_img, 125, cv2.THRESH_BINARY,cv2.THRESH_OTSU)
        canny_img = cv2.Canny(blur_img,40,100)
        seg_img = cv2.erode(canny_img,element)
        #cv2.imshow(wn_gray,canny_img)
    return seg_img
        

# now that segmentaion is done it is time for 
# feature extracting
def colorBin(img):
    # find colors
    color=[[sigma_low_lb_h, sigma_high_lb_h,sigma_low_lb_s, sigma_high_lb_s,sigma_low_lb_v, sigma_high_lb_v],
           [sigma_low_db_h, sigma_high_db_h,sigma_low_db_s, sigma_high_db_s,sigma_low_db_v, sigma_high_db_v],
           [sigma_low_lg_h, sigma_high_lg_h,sigma_low_lg_s, sigma_high_lg_s,sigma_low_lg_v, sigma_high_lg_v],
           [sigma_low_mlg_h, sigma_high_mlg_h,sigma_low_mlg_s, sigma_high_mlg_s,sigma_low_mlg_v, sigma_high_mlg_v],
           [sigma_low_mg_h, sigma_high_mg_h,sigma_low_mg_s, sigma_high_mg_s,sigma_low_mg_v, sigma_high_mg_v],
           [sigma_low_dmg_h, sigma_high_dmg_h,sigma_low_dmg_s, sigma_high_dmg_s,sigma_low_dmg_v, sigma_high_dmg_v],
           [sigma_low_dg_h, sigma_high_dg_h,sigma_low_dg_s, sigma_high_dg_s,sigma_low_dg_v, sigma_high_dg_v]]
    # bins initillaised 
    bin_lB=0
    bin_dB=0
    bin_lG=0
    bin_mlG=0
    bin_mG=0
    bin_mdG=0
    bin_dG=0

    reshp_img = img.reshape((img.shape[0]*img.shape[1],3))
    # fill up the bins
    h, s, v = cv2.split(reshp_img)
    for i in range(len(h)):
        if h[i]<=color[0][0] and h[i]>=color[0][1] and s[i]<=color[0][2]and s[i]>=color[0][3]and v[i]<= color[0][4] and v[i] >=color[0][5]:
            bin_lB +=1

        if h[i]<=color[1][0] and h[i]>=color[1][1] and s[i]<=color[1][2]and s[i]>=color[1][3]and v[i]<= color[1][4] and v[i] >=color[1][5]:
            bin_dB +=1

        if h[i]<=color[2][0] and h[i]>=color[2][1] and s[i]<=color[2][2]and s[i]>=color[2][3]and v[i]<= color[2][4] and v[i] >=color[2][5]:
            bin_lG +=1

        if h[i]<=color[3][0] and h[i]>=color[3][1] and s[i]<=color[3][2]and s[i]>=color[3][3]and v[i]<= color[3][4] and v[i] >=color[3][5]:
            bin_mlG +=1

        if h[i]<=color[4][0] and h[i]>=color[4][1] and s[i]<=color[4][2]and s[i]>=color[4][3]and v[i]<= color[4][4] and v[i] >=color[4][5]:
            bin_mG +=1

        if h[i]<=color[5][0] and h[i]>=color[5][1] and s[i]<=color[5][2]and s[i]>=color[5][3]and v[i]<= color[5][4] and v[i] >=color[5][5]:
            bin_mdG +=1

        if h[i]<=color[6][0] and h[i]>=color[6][1] and s[i]<=color[6][2]and s[i]>=color[6][3]and v[i]<= color[6][4] and v[i] >=color[6][5]:
            bin_dG +=1

    return bin_lB,bin_dB,bin_lG,bin_mlG,bin_mG,bin_mdG,bin_dG

def pixelCount(img):
    reshp_img = img.reshape((img.shape[0]*img.shape[1],1))
    counter = 0 
    for i in range(len(reshp_img)):
        if reshp_img[i] > 0:
            counter +=1
    return counter
# find colors
# find contures

# this function recives a binary image and a colored 
# image, and returns a list of features colorsRGB, colorsHSV,
# perimeter length and area further investigation skeleton features.

def feature_ex(col_img,hsv_img,bin_col,bin_hsv,bin_gray):
    # find colors
    # bins initillaised
    # fill up the bins 
    ligth_brown=0
    dark_brown=0
    light_green=0
    medium_light_green=0
    medium_green=0
    medium_dark_green=0
    dark_green=0
    #ligth_brown,dark_brown,light_green,medium_light_green,medium_green,medium_dark_green,dark_green = colorBin(hsv_img)

    # find countures 
    #col_contours, col_hierarchy = cv2.findContours(bin_col,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #hsv_contours, hsv_hierarchy = cv2.findContours(bin_hsv,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    gray_contours, gray_hierarchy = cv2.findContours(bin_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    print(len(gray_contours))
    print(len(gray_hierarchy))
    for i in range(len(gray_contours)):
        gray_cnt = gray_contours[i]
    
        # area
    
        area_retval = cv2.contourArea(gray_cnt)
        print("area : {}".format(area_retval))
        area_tot=0

        # length
        arc_retval = cv2.arcLength(gray_cnt,True)
        print("arc : {}".format(arc_retval))
        arc_tot=0

    # skeleton
    skel_img = Skeletonizer(col_img)
    skel_tot= pixelCount(skel_img)
    #wn_skel="skel"
    #print(skel_img)
    #cv2.namedWindow(wn_skel, cv2.WINDOW_NORMAL)
    #cv2.imshow(wn_skel,skel_img)
    #cv2.waitKey(0)
    lst_features=[arc_tot,area_tot,skel_tot,ligth_brown,dark_brown,light_green,medium_light_green,medium_green,medium_dark_green,dark_green]
    return lst_features

# now that the features are extracted, the features can be used for 
# either training or evealuting the weed number based on the tranined data
# before this there is a need to save the data to a file that can be used to train on
def save_data(lst,weednumber):

    try:
        sheet = pd.read_execl('traningData.xlsx')
        print("traning file opened")
    except:
        sheet = pd.ExcelWriter('traningData.xlsx')
        print("new file created")
   
    sheet['arc_tot']=lst[0]
    sheet['area_tot']=lst[1]
    sheet['skel_tot']=lst[2]
    sheet['ligth_brown']=lst[3]
    sheet['dark_brown']=lst[4]
    sheet['light_green']=lst[5]
    sheet['medium_light_green']=lst[6]
    sheet['medium_green']=lst[7]
    sheet['medium_dark_green']=lst[8]
    sheet['dark_green']=lst[9]
    sheet.save()

    return

# norm
# classification


# regresion training 
def traning_data():
    pass
# or

# output a weed number on trained data
def weed_number():
    pass

# now the resaspie is determined, it is time to think about loading images 
# into the process 

# getting a path for a image 
# returing a colored, HSV representaion and grayscaled image
def load_image(path):
    img = cv2.imread(path,cv2.IMREAD_COLOR)
    col_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return col_img, gray_img, hsv_img

# lists all dir in the folder
# goes into each folder and listing each file
# from here it is possible to iterate thourgh all 
# images that needs classifying  
def listDir():
    dirNames = os.listdir()
    retval = os.getcwd()
    
    for dirName in dirNames:
        counter = 1
        _dir = os.path.abspath(dirName)
        os.chdir(_dir)
        fileNames = os.listdir()
        for fileName in fileNames:
            #here the functionalties has to be implemented
            counter = counter + 1
            print(counter)         
        os.chdir(retval)

def resipe():
    col_img, hsv_img, gray_img = load_image(" ") 
    
    bin_col_img = segmention(col_img,0)
    bin_hsv_img = segmention(hsv_img,1)
    bin_gray_img = segmention(gray_img,2)

    lst = feature_ex(col_img,hsv_img,bin_col_img,bin_hsv_img,bin_gray_img)

    save_data(lst,weedNumber)


    


def main():
    path = "C:\\Users\\WoomBat\\Aalborg Universitet\\Jonathan Eichild Schmidt - P6 - billeder\\cropped_weednumber_sorted\\10_04282020_01\\011.jpeg"
    col_img, hsv_img, gray_img = load_image(path) # testet done
    #wn_start ="start"
    #cv2.namedWindow(wn_start, cv2.WINDOW_NORMAL)
    #cv2.imshow(wn_start,col_img)
    #cv2.waitKey(0)

    bin_col_img = segmention(col_img,0)# tested done missing fine tuning
    bin_hsv_img = segmention(hsv_img,1)# tested done missing fine tuning
    bin_gray_img = segmention(gray_img,2)# tested done missing fine tuning
    cv2.waitKey(0)
    lst = feature_ex(col_img,hsv_img,bin_col_img,bin_hsv_img,bin_gray_img)# almost done need area and arclengteh colorbin values
    print(lst)
    
    exit(1)


if __name__ == "__main__":
    # execute only if run as a script
    main()    
   