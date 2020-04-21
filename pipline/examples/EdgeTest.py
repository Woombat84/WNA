import cv2
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):
    pass

image = cv2.imread("C:\\Users\\Jonathan\\Source\\Repos\\WNA\\pipline\\examples\\TestBillede.png")
img = cv2.GaussianBlur(image, (9, 9), 0)
canny = cv2.Canny(img, 50, 80)

bit_not = cv2.bitwise_not(canny)
bit_and = cv2.bitwise_and(img, img, mask=bit_not)

img_gray = cv2.cvtColor(bit_and, cv2.COLOR_BGR2GRAY)

#cv2.imshow("Image", img)
#cv2.imshow("Canny", canny)
#cv2.imshow("And", bit_and)

cv2.namedWindow("Trackbars")
cv2.createTrackbar("Thresh", "Trackbars", 10, 255, nothing)

while True:
	thresh = cv2.getTrackbarPos("Thresh", "Trackbars")
	_, threshold = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
	bit_and = cv2.bitwise_and(img, img, mask=threshold)
	
	
	cv2.imshow("Threshold", threshold)
	cv2.imshow("And", bit_and)
	key = cv2.waitKey(100)
	if key == 27:
		break
	

#mean_c = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 9)
#gauss_c = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5)

#ThreshGauss = cv2.bitwise_and(img, img, mask=gauss_c)
#ThreshMean = cv2.bitwise_and(img, img, mask=mean_c)

#cv2.imshow("Image", img)
#cv2.imshow("Canny", canny)
#cv2.imshow("And", bit_and)
#cv2.imshow("Threshold", threshold)

#cv2.imshow("ThresholdMean", mean_c)
#cv2.imshow("ThreshGauss", ThreshGauss)
#cv2.imshow("Threshold", gauss_c)
#cv2.imshow("ThreshMean", ThreshMean)


cv2.waitKey(0)
cv2.destroyAllWindows()