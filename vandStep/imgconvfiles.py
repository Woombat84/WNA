import os
import sys


import cv2  

import numpy as np

def download_image(path):
    
    image = cv2.imread(path,cv2.IMREAD_COLOR)
    return image


def save_image(img,_filename):
    
    crop_img = img[1140:2060,0:2151]
    
    
    cv2.imwrite(_filename,crop_img)
   
    return


def listDir():

    fileNames = os.listdir()
    counter = 32
    for fileName in fileNames:
        save_image(download_image(fileName),fileName)
        counter = counter + 1
        print(counter)


if __name__ == '__main__':
    listDir()
