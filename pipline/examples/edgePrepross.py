import cv2
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):
    pass

image = cv2.imread("C:\\Users\\Jonathan\\Source\\Repos\\WNA\\pipline\\examples\\TestBillede2.png")
cv2.namedWindow("Trackbars")
cv2.createTrackbar("MaxVal", "Trackbars", 80, 255, nothing)
cv2.createTrackbar("MinVal", "Trackbars", 25, 255, nothing)


while True:
    img1 = cv2.GaussianBlur(image, (7, 7), 0)
    img2 = cv2.bilateralFilter(image, 7, 50, 50)

    maxVal = cv2.getTrackbarPos("MaxVal", "Trackbars")
    minVal = cv2.getTrackbarPos("MinVal", "Trackbars")

    canny1 = cv2.Canny(img1, minVal, maxVal)
    canny2 = cv2.Canny(img2, minVal, maxVal)

    bit_not1 = cv2.bitwise_not(canny1)
    bit_not2 = cv2.bitwise_not(canny2)
    bit_and1 = cv2.bitwise_and(img1, img1, mask=bit_not1)
    bit_and2 = cv2.bitwise_and(img2, img2, mask=bit_not2)

    diff = bit_and1 - bit_and2
    #diff = canny1 - canny2

    #cv2.imshow("Image", img)
    cv2.imshow("Canny1", canny1)
    cv2.imshow("And1", bit_and1)
    cv2.imshow("Canny2", canny2)
    cv2.imshow("And2", bit_and2)
    #cv2.imshow("Diff", diff)
    
    key = cv2.waitKey(100)
    if key == 27:
        break

cv2.destroyAllWindows()