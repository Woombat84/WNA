import os
import sys
import cv2  
import numpy as np




def download_image(path):
    
    image = cv2.imread(path,cv2.IMREAD_COLOR)
    return image


def save_image(img,_fileName,_dirName):
    try:
        crop_img = img[0:2150,1140:2060]
        _fileName = "{}.jpeg".format("%03d"%number)
        cv2.imwrite(_fileName,res_img)
    except:
        print("failed at: {}{}".format(_dirName,_fileName))
    return


def listDir():

    dirNames = os.listdir()
    retval = os.getcwd()
    
    for dirName in dirNames:
        if dirName == 'imgconv.py':
                exit(1)
        counter = 1
        _dir = os.path.abspath(dirName)
        os.chdir(_dir)
        fileNames = os.listdir()
        for fileName in fileNames:
            save_image(download_image(fileName),fileName,dirName)
            counter = counter + 1
            print(counter)
         

        os.chdir(retval)


if __name__ == '__main__':
    listDir()
