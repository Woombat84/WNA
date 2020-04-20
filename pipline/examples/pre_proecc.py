import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("C:\\Users\\Jonathan\\Source\\Repos\\WNA\\pipline\\examples\\TestBillede.png")

b, g, r = cv2.split(image)

Green = 2*g-r-b

#print(type(2*g[0][0]))
#print(type(Green[0][0]))


'''
hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

img = cv2.bilateralFilter(image, 11, 150, 150)

# Green color
low_green = np.array([25, 30, 40])
high_green = np.array([88, 255, 255])
green_mask = cv2.inRange(hsv_frame, low_green, high_green)
green = cv2.bitwise_and(image, image, mask=green_mask)


img = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
'''


img = cv2.medianBlur(Green,5)

print(image.shape)

mean_c = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 12)
bit_and = cv2.bitwise_and(image, image, mask=mean_c)

cv2.imshow("img", bit_and)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

