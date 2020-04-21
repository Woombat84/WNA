import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("C:\\Users\\Jonathan\\Source\\Repos\\WNA\\pipline\\examples\\TestBillede.png")
img1 = cv2.GaussianBlur(image, (9, 9), 0)
img2 = cv2.bilateralFilter(image, 7, 50, 50)
canny1 = cv2.Canny(img1, 50, 80)
canny2 = cv2.Canny(img2, 50, 80)

bit_not1 = cv2.bitwise_not(canny1)
bit_not2 = cv2.bitwise_not(canny2)
bit_and1 = cv2.bitwise_and(img1, img1, mask=bit_not1)
bit_and2 = cv2.bitwise_and(img2, img2, mask=bit_not2)

#diff = bit_and1 - bit_and2
#diff = canny1 - canny2

#img_gray = cv2.cvtColor(bit_and, cv2.COLOR_BGR2GRAY)

#cv2.imshow("Image", img)
cv2.imshow("Canny1", canny1)
cv2.imshow("And1", bit_and1)
cv2.imshow("Canny2", canny2)
cv2.imshow("And2", bit_and2)
cv2.imshow("Diff", diff)

cv2.waitKey(0)
cv2.destroyAllWindows()