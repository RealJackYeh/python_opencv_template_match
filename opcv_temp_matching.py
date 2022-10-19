import cv2
import numpy as np
import matplotlib.pyplot as plt
o = cv2.imread('capture.jpg')
img1 = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY) # 灰階影像
t = cv2.imread('template1.jpg')
temp1 = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) # 灰階影像
th, tw = temp1.shape[::]
rv = cv2.matchTemplate(img1, temp1, cv2.TM_SQDIFF)
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(rv)
topLeft = minLoc
bottomRight = (topLeft[0] + tw, topLeft[1] + th)
cv2.rectangle(img1, topLeft, bottomRight, 255, 2)
plt.subplot(1,2,1), plt.imshow(temp1, cmap = 'gray')
plt.title('template')
plt.subplot(1,2,2), plt.imshow(img1, cmap='gray')
plt.title('matching result')
plt.show()