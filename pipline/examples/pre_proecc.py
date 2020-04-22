import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("TestBillede.png")



def weight_sub_img(img,weight_green,weight_red,weight_blue):

    #print(type(2*g[0][0]))
    #print(type(Green[0][0]))

    b, g, r = cv2.split(img)

    weigthed = weight_green*g-weight_red*r-weight_blue*b

    return weigthed



def hsv_img(img):

    hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv = cv2.bilateralFilter(img, 11, 150, 150)

    # Green color
    low_green = np.array([25, 30, 40])
    high_green = np.array([88, 255, 255])
    green_mask = cv2.inRange(hsv_frame, low_green, high_green)
    green = cv2.bitwise_and(image, image, mask=green_mask)

    hsv = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)

    return hsv


img = cv2.medianBlur(hsv_img(image) ,5)

cv2.imshow("img", img)


#mean_c = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 12)

#bit_and = cv2.bitwise_and(image, img, mask=mean_c)

#print(image.shape)

#cv2.imshow("img", bit_and)
key = cv2.waitKey(0)
cv2.destroyAllWindows()



