import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
#filename = 'chessboard/Boards/coba.png'
filename = 'chessboard/Boards/6_edited.jpg'
img = cv.imread(filename)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

edges = cv.Canny(gray,100,300)

houghs = cv.HoughLines(edges,1,np.pi/180,100)

hx = []
hy = []

for hough in houghs:
    rho, theta = hough[0]
    hx.append(int(theta*180/np.pi))
    hy.append(rho)


plt.scatter(hx,hy)
plt.show()

'''
cv.imshow('dst',img)

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
'''