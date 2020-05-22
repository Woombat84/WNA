#!/usr/bin/python3
# This is just a rouhg layout

import cv2
import time
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import openpyxl
import matplotlib.pyplot as plt
import pathlib
import csv
from numpy import genfromtxt
import math

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
def segmention(img):
    # Green color
    low_green = np.array([30, 30, 40])
    high_green = np.array([100, 255, 255])
    green_mask = cv2.inRange(img, low_green, high_green)

    img = cv2.GaussianBlur(green_mask, (9, 9), 0)
    img = cv2.bilateralFilter(img, 7, 100, 100)

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)
    
    kernel2 = np.ones((3,3),np.uint8)
    erosion2 = cv2.erode(erosion,kernel2,iterations = 2)
    
    kernel3 = np.ones((7,7),np.uint8)
    opening = cv2.morphologyEx(erosion2, cv2.MORPH_OPEN, kernel3)

    return opening
        

# now that segmentaion is done it is time for 
# feature extracting
def colorBin(img,lst):
 
    low1 = np.array([0, 30, 30])
    high1 = np.array([10, 255, 255])
    bin1_mask = cv2.inRange(img, low1, high1)
    lst.append(cv2.countNonZero(bin1_mask))

    low2 = np.array([11, 30, 30])
    high2 = np.array([20, 255, 255])
    bin2_mask = cv2.inRange(img, low2, high2)
    lst.append(cv2.countNonZero(bin2_mask))

    low3 = np.array([21, 30, 30])
    high3 = np.array([30, 255, 255])
    bin3_mask = cv2.inRange(img, low3, high3)
    lst.append(cv2.countNonZero(bin3_mask))

    low4 = np.array([41, 30, 30])
    high4 = np.array([50, 255, 255])
    bin4_mask = cv2.inRange(img, low4, high4)
    lst.append(cv2.countNonZero(bin4_mask))

    low5 = np.array([51, 30, 30])
    high5 = np.array([60, 255, 255])
    bin5_mask = cv2.inRange(img, low5, high5)
    lst.append(cv2.countNonZero(bin5_mask))

    low6 = np.array([61, 30, 30])
    high6 = np.array([70, 255, 255])
    bin6_mask = cv2.inRange(img, low6, high6)
    lst.append(cv2.countNonZero(bin6_mask))

    low7 = np.array([71, 30, 30])
    high7 = np.array([80, 255, 255])
    bin7_mask = cv2.inRange(img, low7, high7)
    lst.append(cv2.countNonZero(bin7_mask))

    low8 = np.array([81, 30, 30])
    high8 = np.array([90, 255, 255])
    bin8_mask = cv2.inRange(img, low8, high8)
    lst.append(cv2.countNonZero(bin8_mask))

    low9 = np.array([91, 30, 30])
    high9 = np.array([100, 255, 255])
    bin9_mask = cv2.inRange(img, low9, high9)
    lst.append(cv2.countNonZero(bin9_mask))

    low10 = np.array([101, 30, 30])
    high10 = np.array([110, 255, 255])
    bin10_mask = cv2.inRange(img, low10, high10)
    lst.append(cv2.countNonZero(bin10_mask))

    low11 = np.array([111, 30, 30])
    high11 = np.array([120, 255, 255])
    bin11_mask = cv2.inRange(img, low11, high11)
    lst.append(cv2.countNonZero(bin11_mask))

    low12 = np.array([121, 30, 30])
    high12 = np.array([130, 255, 255])
    bin12_mask = cv2.inRange(img, low12, high12)
    lst.append( cv2.countNonZero(bin12_mask))

    low13 = np.array([131, 30, 30])
    high13 = np.array([140, 255, 255])
    bin13_mask = cv2.inRange(img, low13, high13)
    lst.append(cv2.countNonZero(bin13_mask))

    low14 = np.array([141, 30, 30])
    high14 = np.array([150, 255, 255])
    bin14_mask = cv2.inRange(img, low14, high14)
    lst.append(cv2.countNonZero(bin14_mask))

    low15 = np.array([151, 30, 30])
    high15 = np.array([160, 255, 255])
    bin15_mask = cv2.inRange(img, low15, high15)
    lst.append(cv2.countNonZero(bin15_mask))

    low16 = np.array([161, 30, 30])
    high16 = np.array([170, 255, 255])
    bin16_mask = cv2.inRange(img, low16, high16)
    lst.append(cv2.countNonZero(bin16_mask))

    low17 = np.array([171, 30, 30])
    high17 = np.array([179, 255, 255])
    bin17_mask = cv2.inRange(img, low17, high17)
    lst.append(cv2.countNonZero(bin17_mask))


    return lst

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

def feature_ex(col_img,hsv_img,gray_img,bin_img):
    # find colors
    # fill up the bins 
    lst_features=[]
    kernel = (3,3)
    sigma = 2
    blur_gray = cv2.GaussianBlur(gray_img,kernel,sigma,cv2.BORDER_WRAP)
    lst_features.append(cv2.countNonZero(cv2.Canny(blur_gray,65,255)))
   
    #find countures 
   
    bin_contours, bin_hierarchy = cv2.findContours(bin_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    area_tot=0
    arc_tot=0
    for i in range(len(bin_contours)):
        bin_cnt = bin_contours[i]
    
        # area
        area_retval = cv2.contourArea(bin_cnt)
        #print("area : {}".format(area_retval))
        area_tot+=area_retval

        # length
        arc_retval = cv2.arcLength(bin_cnt,False)
        #print("arc : {}".format(arc_retval))
        arc_tot+=arc_retval
    
    lst_features.append(int(area_tot))
    lst_features.append(int(arc_tot))
    # skeleton
    skel_img = Skeletonizer(col_img)
    lst_features.append(cv2.countNonZero(skel_img))
    lst_features = colorBin(hsv_img,lst_features)
    
    return lst_features

# now that the features are extracted, the features can be used for 
# either training or evealuting the weed number based on the tranined data
# before this there is a need to save the data to a file that can be used to train on
def save_data(lst,path):
    
    filename ="traning_data.csv"
    filepath =os.path.join(path, filename)
    p = pathlib.PureWindowsPath(filepath)
    fp = os.fspath(p)
    
    st =  ""
    for i in range(len(lst)):
        st =st + str(lst[i])
        if i == len(lst)-1:
          st =st + "\n"
        else:
          st =st + ","
    

    with open(p,'a') as nf:
        nf.write(st)

    return


# classification


# regresion training 
def traning_data():
    filename ="traning_data.csv"
    
    lst = genfromtxt(filename, delimiter=',')
    new_lst = np.transpose(lst)
    new_lst = new_lst.astype(int)
    x = new_lst[0:new_lst.shape[0]-1]
    y = new_lst[new_lst.shape[0]-1] 
    #x = [df['arc_tot'],df['skel_tot'],df['ligth_brown'],df['dark_brown'],df['light_green'],df['medium_green'],df['medium_dark_green'],df['dark_green']]
    #y = df['weed_number']
    #x, y = np.array(x), np.array(y)
    x, y = np.array(x.transpose()), np.array(y.transpose())
    min = np.amin(x, axis=0)
    max = np.amax(x, axis=0)
  
    x_norm = np.zeros(x.shape)
    np.seterr(divide='ignore', invalid='ignore')
    for i in range(x.shape[0]):
        x_norm[i] = (x[i]-min)/(max-min)

    x_norm = np.nan_to_num(x_norm)
    X_train, X_test, y_train, y_test = train_test_split(x_norm, y, test_size=0.33, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    r_sq = model.score(x_norm, y) #R^2
    print("R^2: ",r_sq)
    y_pred = model.predict(X_test)
    
    error = 0
    err=np.zeros(y_pred.shape)
    for i in range(y_pred.shape[0]):
        #print("Error: ", y_pred[i]-y_test[i])
        error += math.sqrt((y_pred[i]-y_test[i])**2)
        err[i] = math.sqrt((y_pred[i]-y_test[i])**2)

    error = error/len(y_pred)
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Error' : err})
    minError = np.amin(err)
    maxError = np.amax(err)
    print("Average error: ", error)
    print('Mean:', np.mean(err))
    print('Standard Deviation:', np.std(err))
    strOutput ="Max error: {} Min error: {} Avg error: {} Mean error: {} std: {} R^2: {}".format(maxError,minError,error,np.mean(err),np.std(err),r_sq)
    with open("test_result.txt",'a') as nf:
        nf.write(strOutput)
    df.to_excel("test_data.xlsx")
    np.savetxt('max.npy', max, fmt='%d')
    np.savetxt('min.npy', min, fmt='%d')
    filename = 'WNA_model.sav'
    pickle.dump(model, open(filename, 'wb'))
# or

# output a weed number on trained data
def weed_number(path,model,fullpath):
    col_img, hsv_img, gray_img = load_image(path) # testet done    
    bin_hsv_img = segmention(hsv_img)# tested done missing fine tuning
    lst = feature_ex(col_img,hsv_img,bin_hsv_img)# almost done need colorbin values and test
    x = np.array(lst)
    maxPath =os.path.join(fullpath, 'max.npy')
    minPath =os.path.join(fullpath, 'min.npy')
    max = np.loadtxt(maxPath)
    min = np.loadtxt(minPath)

    x_norm = np.zeros(x.shape)
    np.seterr(divide='ignore', invalid='ignore')
    x_norm = (x-min)/(max-min)
    
    x_norm = np.nan_to_num(x_norm)

    
    WeedNumberScore = model.predict(x_norm)

    filename ="test_data.csv"
    filepath =os.path.join(fullpath, filename)
    p = pathlib.PureWindowsPath(filepath)
    fp = os.fspath(p)
    with open(p,'a') as nf:
       nf.write(WeedNumberScore)

    #result = model.score(X_test, Y_test)
    return WeedNumberScore

# now the resaspie is determined, it is time to think about loading images 
# into the process 

# getting a path for a image 
# returing a colored, HSV representaion and grayscaled image
def load_image(path):
    img = cv2.imread(path,cv2.IMREAD_COLOR)
    col_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
    gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return col_img, hsv_img, gray_img

# lists all dir in the folder
# goes into each folder and listing each file
# from here it is possible to iterate thourgh all 
# images that needs classifying  
def listDir(model="" ,training=True):
    dirNames = os.listdir()
    
    counter = 0
    for dirName in dirNames:
        
        retval = os.getcwd()
        
        dirSplit = str(dirName).split('_')
        weedNumber= dirSplit[0]
        if os.path.isdir(dirName):
            os.chdir(dirName)
            
            fileNames = os.listdir()
            for fileName in fileNames:
                currentdir=os.getcwd()
                print("{}\{}".format(currentdir,fileName))
                if training == True:
                    resipe(fileName,weedNumber,retval)
                if training == False:
                    weed_number(fileName,model,retval)
                counter = counter + 1
                print(counter)         
            os.chdir(retval)
    return


def resipe(path,weedNumber,fullpath):
    
 
    col_img, hsv_img, gray_img = load_image(path) # testet done    
    bin_hsv_img = segmention(hsv_img)# tested done missing fine tuning
    lst = feature_ex(col_img,hsv_img,gray_img,bin_hsv_img)# almost done need colorbin values and test
    lst.append(weedNumber)
    save_data(lst,fullpath)


def main():
    
    
    # when ready release next lines
    
    t1=input('training? y for yes n for no any thing else to skip:')
    if t1 == 'y':
       listDir()
    if t1 == 'n':
       filename = 'WNA_model.sav'
       model = pickle.load(open(filename, 'rb'))
       listDir(model,False)
    
    t2=input('are we ready to create a model? y for yes n for no :')
    if t2 =='y':
       traning_data() 

    exit(1)


if __name__ == "__main__":
    # execute only if run as a script
    main()    
   
    #missing parts for now
   
    
    
