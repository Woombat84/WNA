import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("C:\\Users\\Jonathan\\Source\\Repos\\WNA\\pipline\\examples\\TestBillede.png")
img = cv2.GaussianBlur(image, (15, 15), 0)
canny = cv2.Canny(img, 50, 80)

bit_not = cv2.bitwise_not(canny)
bit_and = cv2.bitwise_and(img, img, mask=bit_not)

cv2.imshow("Image", img)
cv2.imshow("Canny", canny)
cv2.imshow("And", bit_and)


cv2.waitKey(0)
cv2.destroyAllWindows()