#!/usr/bin/python3
# This is just a rouhg layout

import cv2
import time

import numpy as np
#from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

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
def segmention(img):
    # Green color
    low_green = np.array([30, 30, 40])
    high_green = np.array([100, 255, 255])
    green_mask = cv2.inRange(img, low_green, high_green)

    img = cv2.GaussianBlur(green_mask, (9, 9), 0)
    img = cv2.bilateralFilter(img, 7, 100, 100)

    Equal = cv2.equalizeHist(img[:,:])


    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)

    kernel2 = np.ones((3,3),np.uint8)
    erosion2 = cv2.erode(erosion,kernel2,iterations = 2)

    kernel3 = np.ones((7,7),np.uint8)
    opening = cv2.morphologyEx(erosion2, cv2.MORPH_OPEN, kernel3)

    return opening
        

# now that segmentaion is done it is time for 
# feature extracting
def colorBin(img):
 
    low_bin_lB = np.array([140, 18, 53])
    high_bin_lB = np.array([33, 31, 66])
    bin_lB_mask = cv2.inRange(img, low_bin_lB, high_bin_lB)
    bin_lB = cv2.countNonZero(bin_lB_mask)

    low_bin_dB = np.array([90,15,68])
    high_bin_dB = np.array([223,24,82])
    bin_dB_mask = cv2.inRange(img, low_bin_dB, high_bin_dB)
    bin_dB = cv2.countNonZero(bin_dB_mask)

    low_bin_lG = np.array([41,146,106])
    high_bin_lG = np.array([52,188,196])
    bin_lG_mask = cv2.inRange(img, low_bin_lG, high_bin_lG)
    bin_lG = cv2.countNonZero(bin_lG_mask)

    low_bin_mG = np.array([56,110,66])
    high_bin_mG = np.array([63,193,97])
    bin_mG_mask = cv2.inRange(img, low_bin_mG, high_bin_mG)
    bin_mG = cv2.countNonZero(bin_mG_mask)
 
    low_bin_mdG = np.array([57,140,95])
    high_bin_mdG = np.array([60,171,117])
    bin_mdG_mask = cv2.inRange(img, low_bin_mdG, high_bin_mdG)
    bin_mdG = cv2.countNonZero(bin_mdG_mask)

    low_bin_dG = np.array([30,41,83])
    high_bin_dG = np.array([43,63,120])
    bin_dG_mask = cv2.inRange(img, low_bin_dG, high_bin_dG)
    bin_dG = cv2.countNonZero(bin_dG_mask)
    
    
    return bin_lB,bin_dB,bin_lG,bin_mG,bin_mdG,bin_dG

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

def feature_ex(col_img,hsv_img,bin_img):
    # find colors
    # fill up the bins 

    ligth_brown,dark_brown,light_green,medium_green,medium_dark_green,dark_green = colorBin(hsv_img)
   
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
    
    
    # skeleton
    skel_img = Skeletonizer(col_img)
    skel_tot = pixelCount(skel_img)
    
    lst_features=[arc_tot,area_tot,skel_tot,ligth_brown,dark_brown,light_green,medium_green,medium_dark_green,dark_green]
    return lst_features

# now that the features are extracted, the features can be used for 
# either training or evealuting the weed number based on the tranined data
# before this there is a need to save the data to a file that can be used to train on
def save_data(lst,weednumber):

    try:
      sheet = pd.read_excel("./traning_data.xlsx",index_col=0)
      #print("traning file opened")
      #print(sheet.head())
      new_data=pd.DataFrame({'arc_tot': lst[0],
      'area_tot': lst[1],
      'skel_tot':lst[2],
      'ligth_brown':lst[3],
      'dark_brown':lst[4],
      'light_green':lst[5],
      'medium_green': lst[6],
      'medium_dark_green':lst[7],
      'dark_green':lst[8],
      'weed_number':[weednumber]})
      new_sheet=sheet.append(new_data, ignore_index=True)
      #print(new_sheet.head())
      writer = pd.ExcelWriter("./traning_data.xlsx",engine='openpyxl',index=False)
      new_sheet.to_excel(writer)
      writer.save()
    except:
      sheet = pd.ExcelWriter('traningData.xlsx')
      print("new file created")
      new_data=pd.DataFrame({'arc_tot':[lst[0]],
      'area_tot': [lst[1]],
      'skel_tot':[lst[2]],
      'ligth_brown':[lst[3]],
      'dark_brown':[lst[4]],
      'light_green':[lst[5]],
      'medium_green':[lst[7]],
      'medium_dark_green':[lst[8]],
      'dark_green':[lst[9]],
      'weed_number':[weednumber]})
      writer = pd.ExcelWriter("./traning_data.xlsx",engine='openpyxl',index=False)
      new_data.to_excel(writer)
      writer.save()

    return


# classification


# regresion training 
def traning_data():
    file = "./traning_data.xlsx"
    df = pd.read_excel(file)

    print(df)

    x = [df['arc_tot'],df['area_tot'],df['skel_tot'],df['ligth_brown'],df['dark_brown'],df['light_green'],df['medium_light_green'],df['medium_green'],df['medium_dark_green'],df['dark_green']]
    y = df['weed_number']
    # norm
    x, y = np.array(x), np.array(y)

    x = x.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

    model = LinearRegression().fit(X_train, y_train)
    r_sq = model.score(x, y) #R^2

    print("R^2: ",r_sq)

    y_pred = model.predict(X_test)

    print("Predicted values: ", y_pred)
    print("Actual values: ", y_test)

    filename = 'WNA_model.sav'
    pickle.dump(model, open(filename, 'wb'))
# or

# output a weed number on trained data
def weed_number(path,model):
    col_img, hsv_img, gray_img = load_image(path) 
    bin_hsv_img = segmention(hsv_img)
    lst = feature_ex(col_img,hsv_img,bin_col_img,bin_hsv_img,bin_gray_img)
    x = [lst[0],lst[1],lst[2],lst[3],lst[4],lst[5],lst[6],lst[7],lst[8]]
    X = np.array(x)
    x = x.reshape(-1, 1)
    WeedNumber = model.predict(X)
    #result = model.score(X_test, Y_test)
    return WeedNumber

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
def listDir(model,training=True):
    dirNames = os.listdir()
    retval = os.getcwd()
    
    for dirName in dirNames:
        counter = 1
        _dir = os.path.abspath(dirName)
        dirSplit = str(dirName).split('_')
        weedNumber= dirSplit[0]
        os.chdir(dirName)
        fileNames = os.listdir()
        for fileName in fileNames:
            if training == True:
                resipe(fileName,weedNumber)
            if training == False:
                weed_number(fileName,model)
            counter = counter + 1
            print(counter)         
        os.chdir(retval)

def resipe(path,weedNumber):
    
    col_img, hsv_img, gray_img = load_image(path) 
    bin_col_img = segmention(col_img,0)
    bin_hsv_img = segmention(hsv_img,1)
    bin_gray_img = segmention(gray_img,2)
    lst = feature_ex(col_img,hsv_img,bin_col_img,bin_hsv_img,bin_gray_img)
    save_data(lst,weedNumber)


def main():
    path = "C:\\Users\\WoomBat\\Aalborg Universitet\\Jonathan Eichild Schmidt - P6 - billeder\\cropped_weednumber_sorted\\10_04282020_01\\011.jpeg"
    col_img, hsv_img, gray_img = load_image(path) # testet done    
    bin_hsv_img = segmention(hsv_img)# tested done missing fine tuning
    lst = feature_ex(col_img,hsv_img,bin_hsv_img)# almost done need colorbin values and test
    print(lst)
    # when ready release next lines
    # t1 = input(training? y for yes n for no :)
    # if == 'y':
    #   filename = 'WNA_model.sav'
    #   model = pickle.load(open(filename, 'rb'))
    #   listDir(model,True)
    
    # t2 = input(are we ready to create a model? y for yes n for no :)
    #if t2 =='y':
    #   traning_data() 

    exit(1)


if __name__ == "__main__":
    # execute only if run as a script
    main()    
   
    #missing parts for now
   
    
    