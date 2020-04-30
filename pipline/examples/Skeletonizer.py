import cv2
import numpy as np


def Skeletonizer(img):

    if img.shape[2] != 1:
       img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)

    ret,img = cv2.threshold(img,127,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    
    while( not done):
        print("here")
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
    

    cv2.imshow("skel",skel)
    cv2.waitKey(0)
    return skel 

img = cv2.imread("C:\\Users\\Jonathan\\Source\\Repos\\WNA\\pipline\\examples\\TestBillede.png")
Skeletonizer(img)