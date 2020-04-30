#!/usr/bin/python3
#This is just a rouhg layout

import cv2
import time

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd


import matplotlib.pyplot as plt



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
    if color == 0:
        blur_img = cv2.GaussianBlur(img, (9, 9), 0)
        canny = cv2.Canny(blur_img, 50, 80)

    
    if color == 1:
        blur_img = cv2.GaussianBlur(img, (9, 9), 0)
        canny = cv2.Canny(blur_img, 50, 80)
        
    if color == 2:
        blur_img = cv2.GaussianBlur(img, (9, 9), 0)
        retval, seg_img = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
    
    return seg_img
        

# now that segmentaion is done it is time for 
# feature extracting


# find colors
# find contures

# this function recives a binary image and a colored 
# image, and returns a list of features colorsRGB, colorsHSV,
# perimeter length and area further investigation skeleton features.

def feature_ex(img,hsv,gray):
    # find colors

    # find countures 
    col_contours, col_hierarchy	= cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    hsv_contours, hsv_hierarchy	= cv2.findContours(hsv,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    gray_contours, gray_hierarchy = cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)



    return lst_features




# now that the features are extracted, the features can be used for 
# either training or evealuting the weed number based on the tranined data
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
    col_img = cv2.imread(path,cv2.IMREAD_COLOR)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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
    
    bi_col_img = segmention(col_img,0)
    bi_hsv_img = segmention(hsv_img,1)
    bi_gray_img = segmention(gray_img,2)


def main():
    path = "C:\\Users\\WoomBat\\Aalborg Universitet\\Jonathan Eichild Schmidt - P6 - billeder\\cropped_weednumber_sorted\\10_04282020_01\\011.jpeg"
    col_img, hsv_img, gray_img = load_image(path) 
    
    bi_col_img = segmention(col_img,0)

    
    
    exit(1)


if __name__ == "__main__":
    # execute only if run as a script
    main()    
   