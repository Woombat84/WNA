import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("C:\\Users\\Jonathan\\Source\\Repos\\WNA\\pipline\\examples\\TestBillede.png")

hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

img = cv2.medianBlur(hsv_frame,11)

# Green color
low_green = np.array([25, 30, 40])
high_green = np.array([88, 255, 255])
green_mask = cv2.inRange(hsv_frame, low_green, high_green)
green = cv2.bitwise_and(image, image, mask=green_mask)

img = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)

img = cv2.medianBlur(img,5)

cv2.imshow("img", img)
key = cv2.waitKey(0)

