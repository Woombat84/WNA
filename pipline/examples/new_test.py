#!/usr/bin/python3
#This is just a rouhg layout

import cv2
import time

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd


import matplotlib.pyplot as plt



#image aqustion and pre-prosseccing is now done it is time for 
#segmentation

#bluring
#thresholding on color hsv
def segmention(img):
    
    
    return seg_img
        

#now that segmentaion is done it is time for 
#feature extracting


#edge detection 
#contures

# this function recives a binary image and a colored 
# image, and returns a list of features colors, perimeter length and area.

def feature_ex():



    return lst_features

#now that the features are extracted, the features can be used for 
#either training or evealuting the weed number based on the tranined data
#classification


#regresion training 
def traning_data():
    pass
#or

# output a weed number on trained data
def weed_number():
    pass
#now the resaspie is determined, it is time to think about loading images 
#into the process 

#getting a path for a image 
#returing a colored image
def load_image(path):
    img = cv2.imread(path,cv2.IMREAD_COLOR)
    return img



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

def main():
    
    segmention()
    exit(1)

if __name__ == "__main__":
    # execute only if run as a script
    main()    
   