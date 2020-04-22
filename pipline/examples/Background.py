import cv2

'''
## For creating a uniform background image for subtraction
img = cv2.imread("C:\\Users\\Jonathan\\Source\\Repos\\WNA\\pipline\\examples\\Background.png")
img2 =img.copy()
retval = cv2.mean(img)

rows,cols, ch = img.shape

for i in range(rows):
    for j in range(cols):
       img2[i,j]= retval[0:3]

#cv2.imshow("Mean", img2)
cv2.imwrite("./Background2.png", img2)
'''

image = cv2.imread("C:\\Users\\Jonathan\\Source\\Repos\\WNA\\pipline\\examples\\TestBillede.png")
background = cv2.imread("C:\\Users\\Jonathan\\Source\\Repos\\WNA\\pipline\\examples\\Background2.png")
retval = cv2.mean(background)
B = int(round(retval[0]))
G = int(round(retval[1]))
R = int(round(retval[2]))

img = image.copy()
rows,cols, ch = img.shape

for i in range(rows):
    for j in range(cols):
       img[i,j] = (B,G,R)


dst	= cv2.subtract(image,img)

cv2.imshow("Original", image)
cv2.imshow("Subtraction", dst)

cv2.waitKey(0)

