import cv2
import numpy as np

img = cv2.imread("./image.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

img = cv2.GaussianBlur(img, (9, 9), 0)
img = cv2.bilateralFilter(img, 7, 100, 100)

canny_img = cv2.Canny(img,40,100)
cv2.imwrite("./ReportImages/Canny0.png", canny_img)

blur = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
cv2.imwrite("./ReportImages/blurring1.png", blur)

# Green color
low_green = np.array([20, 30, 30])
high_green = np.array([100, 255, 255])
green_mask = cv2.inRange(img, low_green, high_green)

NOT = cv2.bitwise_not(green_mask)
AND = cv2.bitwise_and(blur, blur, mask=NOT)
ANDNOT = cv2.bitwise_and(blur, blur, mask=green_mask)

cv2.imwrite("./ReportImages/AND.png", AND)
cv2.imwrite("./ReportImages/ANDNOT.png", ANDNOT)
cv2.imwrite("./ReportImages/GreenMask2.png", green_mask)

kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(green_mask,kernel,iterations = 1)

kernel2 = np.ones((3,3),np.uint8)
erosion2 = cv2.erode(erosion,kernel2,iterations = 1)

kernel3 = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(erosion2, cv2.MORPH_OPEN, kernel3)

cv2.imwrite("./ReportImages/Opening3.png",opening)
