import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

filename = 'chessboard/Boards/6_edited.jpg'
img = cv.imread(filename)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

edges = cv.Canny(gray,100,300)

houghs = cv.HoughLines(edges,1,np.pi/180,100)

thetaValues = []
rhoValues = []

for hough in houghs:
    rho, theta = hough[0]
    thetaValues.append(int(theta*180/np.pi))
    rhoValues.append(rho)


plt.figure(1)
plt.scatter(hx, hy)
plt.figure(2)
plt.imshow(cv.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()