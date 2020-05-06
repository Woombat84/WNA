import numpy as np
import cv2

def colorBin(img):
 
    low1 = np.array([0, 30, 30])
    high1 = np.array([10, 255, 255])
    bin1_mask = cv2.inRange(img, low1, high1)
    bin1 = cv2.countNonZero(bin1_mask)

    low2 = np.array([11, 30, 30])
    high2 = np.array([20, 255, 255])
    bin2_mask = cv2.inRange(img, low2, high2)
    bin2 = cv2.countNonZero(bin2_mask)

    low3 = np.array([21, 30, 30])
    high3 = np.array([30, 255, 255])
    bin3_mask = cv2.inRange(img, low3, high3)
    bin3 = cv2.countNonZero(bin3_mask)

    low4 = np.array([41, 30, 30])
    high4 = np.array([50, 255, 255])
    bin4_mask = cv2.inRange(img, low4, high4)
    bin4 = cv2.countNonZero(bin4_mask)

    low5 = np.array([51, 30, 30])
    high5 = np.array([60, 255, 255])
    bin5_mask = cv2.inRange(img, low5, high5)
    bin5 = cv2.countNonZero(bin5_mask)

    low6 = np.array([61, 30, 30])
    high6 = np.array([70, 255, 255])
    bin6_mask = cv2.inRange(img, low6, high6)
    bin6 = cv2.countNonZero(bin6_mask)

    low7 = np.array([71, 30, 30])
    high7 = np.array([80, 255, 255])
    bin7_mask = cv2.inRange(img, low7, high7)
    bin7 = cv2.countNonZero(bin7_mask)

    low8 = np.array([81, 30, 30])
    high8 = np.array([90, 255, 255])
    bin8_mask = cv2.inRange(img, low8, high8)
    bin8 = cv2.countNonZero(bin8_mask)

    low9 = np.array([91, 30, 30])
    high9 = np.array([100, 255, 255])
    bin9_mask = cv2.inRange(img, low9, high9)
    bin9 = cv2.countNonZero(bin9_mask)

    low10 = np.array([101, 30, 30])
    high10 = np.array([110, 255, 255])
    bin10_mask = cv2.inRange(img, low10, high10)
    bin10 = cv2.countNonZero(bin10_mask)

    low11 = np.array([111, 30, 30])
    high11 = np.array([120, 255, 255])
    bin11_mask = cv2.inRange(img, low11, high11)
    bin11 = cv2.countNonZero(bin11_mask)

    low12 = np.array([121, 30, 30])
    high12 = np.array([130, 255, 255])
    bin12_mask = cv2.inRange(img, low12, high12)
    bin12 = cv2.countNonZero(bin12_mask)

    low13 = np.array([131, 30, 30])
    high13 = np.array([140, 255, 255])
    bin13_mask = cv2.inRange(img, low13, high13)
    bin13 = cv2.countNonZero(bin13_mask)

    low14 = np.array([141, 30, 30])
    high14 = np.array([150, 255, 255])
    bin14_mask = cv2.inRange(img, low14, high14)
    bin14 = cv2.countNonZero(bin14_mask)

    low15 = np.array([151, 30, 30])
    high15 = np.array([160, 255, 255])
    bin15_mask = cv2.inRange(img, low15, high15)
    bin15 = cv2.countNonZero(bin15_mask)

    low16 = np.array([161, 30, 30])
    high16 = np.array([170, 255, 255])
    bin16_mask = cv2.inRange(img, low16, high16)
    bin16 = cv2.countNonZero(bin16_mask)

    low17 = np.array([171, 30, 30])
    high17 = np.array([179, 255, 255])
    bin17_mask = cv2.inRange(img, low17, high17)
    bin17 = cv2.countNonZero(bin17_mask)

    return bin1, bin2, bin3, bin4, bin5, bin6, bin7, bin8, bin9, bin10, bin11, bin12, bin13, bin14, bin15, bin16, bin17