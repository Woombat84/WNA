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




'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("C:\\Users\\Jonathan\\Source\\Repos\\WNA\\pipline\\examples\\TestBillede2.png")
img1 = cv2.GaussianBlur(image, (7, 7), 0)
img2 = cv2.bilateralFilter(image, 7, 50, 50)
canny1 = cv2.Canny(img1, 70, 100)
canny2 = cv2.Canny(img2, 50, 80)

bit_not1 = cv2.bitwise_not(canny1)
bit_not2 = cv2.bitwise_not(canny2)
bit_and1 = cv2.bitwise_and(img1, img1, mask=bit_not1)
bit_and2 = cv2.bitwise_and(img2, img2, mask=bit_not2)


lines = cv2.HoughLinesP(canny1, 1, np.pi/180, 30, maxLineGap=5)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img1, (x1, y1), (x2, y2), (0, 0, 255), 3)

cv2.imshow("Edges", canny1)
cv2.imshow("Image", img1)
cv2.imshow("Normal", image)

#diff = bit_and1 - bit_and2
#diff = canny1 - canny2

#img_gray = cv2.cvtColor(bit_and, cv2.COLOR_BGR2GRAY)

#cv2.imshow("Image", img)
#cv2.imshow("Canny1", canny1)
#cv2.imshow("And1", bit_and1)
#cv2.imshow("Canny2", canny2)
#cv2.imshow("And2", bit_and2)
#cv2.imshow("Diff", diff)

cv2.waitKey(0)
cv2.destroyAllWindows()

'''