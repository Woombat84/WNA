import cv2
import numpy as np
from matplotlib import pyplot as plt

col_image = cv2.imread("TestBillede.png",cv2.IMREAD_COLOR)
gray_image = cv2.cvtColor(col_image, cv2.COLOR_BGR2GRAY)
hsv_image = cv2.cvtColor(col_image, cv2.COLOR_BGR2HSV)

wn_color = "color"
wn_gray = "gray"
wn_hsv ="hsv"

cv2.namedWindow(wn_color, cv2.WINDOW_NORMAL) 
cv2.namedWindow(wn_gray, cv2.WINDOW_NORMAL) 
cv2.namedWindow(wn_hsv, cv2.WINDOW_NORMAL)


col_retval, col_dst = cv2.threshold(col_image, 255, cv2.THRESH_BINARY, cv2.THRESH_TRUNC)
gray_retval, gray_dst = cv2.threshold(gray_image, 255,cv2.THRESH_BINARY, cv2.THRESH_OTSU)
hsv_retval, hsv_dst = cv2.threshold(hsv_image, 255, cv2.THRESH_BINARY, cv2.THRESH_TRUNC)


wn_pre_blur_can_color = "pre_blur_can_color"
wn_pre_blur_can_gray = "pre_blur_can_gray"
wn_pre_blur_can_hsv = "pre_blur_can_hsv"

cv2.namedWindow(wn_pre_blur_can_color, cv2.WINDOW_NORMAL) 
cv2.namedWindow(wn_pre_blur_can_gray, cv2.WINDOW_NORMAL) 
cv2.namedWindow(wn_pre_blur_can_hsv, cv2.WINDOW_NORMAL)

#canny pre blur

col_pre_intv = 10
gray_pre_intv = 10
hsv_pre_intv = 10

col_pre_edges = 0.1
gray_pre_edges = 0.1
hsv_pre_edges = 0.1

col_pre_can = cv2.Canny(col_image,col_retval-col_pre_intv,col_retval+col_pre_intv)
gray_pre_can = cv2.Canny(gray_image,gray_retval-gray_pre_intv,gray_retval+gray_pre_intv)
hsv_pre_can = cv2.Canny(hsv_image,hsv_retval-hsv_pre_intv,hsv_retval+hsv_pre_intv)


#bluring


wn_post_blur_can_color = "post_blur_can_color"
wn_post_blur_can_gray = "post_blur_can_gray"
wn_post_blur_can_hsv = "post_blur_can_hsv"

cv2.namedWindow(wn_post_blur_can_color, cv2.WINDOW_NORMAL) 
cv2.namedWindow(wn_post_blur_can_gray, cv2.WINDOW_NORMAL) 
cv2.namedWindow(wn_post_blur_can_hsv, cv2.WINDOW_NORMAL)


kernel = (9,9)
sigma = 0
blur_col = cv2.GaussianBlur(col_image,kernel,sigma,cv2.BORDER_WRAP)
blur_gray = cv2.GaussianBlur(gray_image,kernel,sigma,cv2.BORDER_WRAP)
blur_hsv = cv2.GaussianBlur(hsv_image,kernel,sigma,cv2.BORDER_WRAP)

#new threshold

col_blur_retval, col_blur_dst = cv2.threshold(col_image, 255, cv2.THRESH_BINARY, cv2.THRESH_TRUNC)
gray_blur_retval, gray_blur_dst = cv2.threshold(gray_image, 255,cv2.THRESH_BINARY, cv2.THRESH_OTSU)
hsv_blur_retval, hsv_blur_dst = cv2.threshold(hsv_image, 255, cv2.THRESH_BINARY, cv2.THRESH_TRUNC)


#canny

col_post_intv = 10
gray_post_intv = 10
hsv_post_intv = 10


col_post_can = cv2.Canny(blur_col,col_blur_retval-col_post_intv,col_blur_retval+col_post_intv)
gray_post_can = cv2.Canny(blur_gray,gray_blur_retval-gray_post_intv,gray_blur_retval+gray_post_intv)
hsv_post_can = cv2.Canny(blur_hsv,hsv_blur_retval-hsv_post_intv,hsv_blur_retval+hsv_post_intv)

#


cv2.imshow(wn_color,col_image)
cv2.imshow(wn_gray,gray_image)
cv2.imshow(wn_hsv,hsv_image)

cv2.imshow(wn_pre_blur_can_color,col_pre_can)
cv2.imshow(wn_pre_blur_can_gray,gray_pre_can)
cv2.imshow(wn_pre_blur_can_hsv,hsv_pre_can)

cv2.imshow(wn_post_blur_can_color,col_post_can)
cv2.imshow(wn_post_blur_can_gray,gray_post_can)
cv2.imshow(wn_post_blur_can_hsv,hsv_post_can)


cv2.waitKey(0)

cv2.destroyAllWindows()