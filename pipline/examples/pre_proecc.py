import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("C:\\Users\\Jonathan\\Source\\Repos\\WNA\\pipline\\examples\\TestBillede.png")

cv2.imshow("img", img)
key = cv2.waitKey(0)

