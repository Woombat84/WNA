import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

def plotting(colshcem,img,name):
    
    p = plt.figure(name)
    
    if colshcem == 0:
        b, g, r = cv2.split(img)
        plt.subplot(311)
        plt.hist(b.ravel(), 256, [0, 256], color = 'blue')
        plt.figlegend('blue')
        plt.subplot(312)        
        plt.hist(g.ravel(), 256, [0, 256], color = 'green')
        plt.figlegend('green')
        plt.subplot(313)
        plt.hist(r.ravel(), 256, [0, 256], color = 'red')
        plt.figlegend('red')
    if colshcem == 1:
        h, s, v = cv2.split(img)
        plt.subplot(311)
        plt.hist(h.ravel(), 256, [0, 256], color = 'blue')
        plt.figlegend('hue')
        plt.subplot(312)        
        plt.hist(s.ravel(), 256, [0, 256], color = 'green')
        plt.figlegend('saturation')
        plt.subplot(313)
        plt.hist(v.ravel(), 256, [0, 256], color = 'red')
        plt.figlegend('value')
    if colshcem == 2:
        plt.subplot(311)
        plt.hist(img.ravel(), 256, [0, 256], color = 'gray')
        plt.figlegend('gray scale value')
        
    
    time.sleep(1)
    return p

col_image = cv2.imread("TestBillede.png",cv2.IMREAD_COLOR)
gray_image = cv2.cvtColor(col_image, cv2.COLOR_BGR2GRAY)
hsv_image = cv2.cvtColor(col_image, cv2.COLOR_BGR2HSV)

col_hist = plotting(0,col_image,"color histogram")
hsv_hist = plotting(1,hsv_image,"hsv histogram")
gray_hist = plotting(2,gray_image,"gray histogram")

plt.savefig("color histogram")
plt.savefig("hsv histogram")
plt.savefig("gray histogram")

wn_color = "color"
wn_gray = "gray"
wn_hsv ="hsv"

cv2.namedWindow(wn_color, cv2.WINDOW_NORMAL) 
cv2.namedWindow(wn_gray, cv2.WINDOW_NORMAL) 
cv2.namedWindow(wn_hsv, cv2.WINDOW_NORMAL)

col_retval, col_dst = cv2.threshold(col_image, 255, cv2.THRESH_BINARY, cv2.THRESH_TRUNC)
gray_retval, gray_dst = cv2.threshold(gray_image, 255,cv2.THRESH_BINARY, cv2.THRESH_OTSU)
hsv_retval, hsv_dst = cv2.threshold(hsv_image, 255, cv2.THRESH_BINARY, cv2.THRESH_TRUNC)

col_hist = plotting(0,col_image,"pre blur color histogram")
hsv_hist = plotting(1,hsv_image,"pre blur hsv histogram")
gray_hist = plotting(2,gray_image,"pre blur gray histogram")

plt.savefig("pre blur color histogram")
plt.savefig("pre blur hsv histogram")
plt.savefig("pre blur gray histogram")

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

wn_pre_blur_inc_can_color = "pre_blur_inc_can_color"
wn_pre_blur_inc_can_gray = "pre_blur_inc_can_gray"
wn_pre_blur_inc_can_hsv = "pre_blur_inc_can_hsv"

bit_not1 = cv2.bitwise_not(col_pre_can)
bit_not2 = cv2.bitwise_not(gray_pre_can)
bit_not3 = cv2.bitwise_not(gray_pre_can)    
bit_and1 = cv2.bitwise_and(col_image, col_image, mask=bit_not1)
bit_and2 = cv2.bitwise_and(hsv_image, hsv_image, mask=bit_not2)
bit_and3 = cv2.bitwise_and(gray_image, gray_image, mask=bit_not3)

wn_pre_blur_ex_can_color = "pre_blur_ex_can_color"
wn_pre_blur_ex_can_gray = "pre_blur_ex_can_gray"
wn_pre_blur_ex_can_hsv = "pre_blur_ex_can_hsv"

#bluring

wn_post_blur_can_color = "post_blur_can_color"
wn_post_blur_can_gray = "post_blur_can_gray"
wn_post_blur_can_hsv = "post_blur_can_hsv"

cv2.namedWindow(wn_post_blur_can_color, cv2.WINDOW_NORMAL) 
cv2.namedWindow(wn_post_blur_can_gray, cv2.WINDOW_NORMAL) 
cv2.namedWindow(wn_post_blur_can_hsv, cv2.WINDOW_NORMAL)

kernel = (3,3)
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

col_hist = plotting(0,col_image,"post blur color histogram")
hsv_hist = plotting(1,hsv_image,"post blur hsv histogram")
gray_hist = plotting(2,gray_image,"post blur gray histogram")

plt.savefig("post blur color histogram")
plt.savefig("post blur hsv histogram")
plt.savefig("post blur gray histogram")

col_post_can = cv2.Canny(blur_col,col_blur_retval-col_post_intv,col_blur_retval+col_post_intv)
gray_post_can = cv2.Canny(blur_gray,gray_blur_retval-gray_post_intv,gray_blur_retval+gray_post_intv)
hsv_post_can = cv2.Canny(blur_hsv,hsv_blur_retval-hsv_post_intv,hsv_blur_retval+hsv_post_intv)

#image showing

cv2.imshow(wn_color,col_image)
cv2.imshow(wn_gray,gray_image)
cv2.imshow(wn_hsv,hsv_image)

cv2.imshow(wn_pre_blur_can_color,col_pre_can)
cv2.imshow(wn_pre_blur_can_gray,gray_pre_can)
cv2.imshow(wn_pre_blur_can_hsv,hsv_pre_can)

cv2.imshow(wn_pre_blur_inc_can_color,bit_and1)
cv2.imshow(wn_pre_blur_inc_can_gray,bit_and3)
cv2.imshow(wn_pre_blur_inc_can_hsv,bit_and2)

cv2.imshow(wn_post_blur_can_color,col_post_can)
cv2.imshow(wn_post_blur_can_gray,gray_post_can)
cv2.imshow(wn_post_blur_can_hsv,hsv_post_can)

cv2.waitKey(0)

cv2.destroyAllWindows()