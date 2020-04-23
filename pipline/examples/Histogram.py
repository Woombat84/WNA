import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("C:\\Users\\Jonathan\\Source\\Repos\\WNA\\pipline\\examples\\TestBillede.png")
b, g, r = cv2.split(img)
    
cv2.imshow("img", img)
#cv2.imshow("b", b)
#cv2.imshow("g", g)
#cv2.imshow("r", r)
    
plt.figure(1)
    
plt.subplot(311)
plt.hist(b.ravel(), 256, [0, 256], color = 'blue')
   
plt.subplot(312)
plt.hist(g.ravel(), 256, [0, 256], color = 'green')
    
plt.subplot(313)
plt.hist(r.ravel(), 256, [0, 256], color = 'red')

plt.show()