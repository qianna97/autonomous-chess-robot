import cv2
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import normalize

def nothing(x):
    pass

def createBar():
    cv2.namedWindow("Edges n Lines")
    cv2.createTrackbar("Min Edges","Edges n Lines",100,1000,nothing)
    cv2.createTrackbar("Max Edges","Edges n Lines",200,1000,nothing)
    cv2.createTrackbar("threshold","Edges n Lines",87,1000,nothing)
    cv2.createTrackbar("con","Edges n Lines",1000,1000,nothing)

def getBar(item,win):
    return cv2.getTrackbarPos(item,win)

def find_intersection(l1,l2):
    a1 = [l1[0],l1[1]] 
    a2 = [l1[2],l1[3]]

    b1 = [l2[0],l2[1]] 
    b2 = [l2[2],l2[3]]

    s = np.vstack([a1,a2,b1,b2])
    h = np.hstack((s, np.ones((4, 1)))) 
    l1 = np.cross(h[0], h[1])           
    l2 = np.cross(h[2], h[3])           
    x, y, z = np.cross(l1, l2)          
    if z == 0:                         
        return (False, False)
    return (int(x/z), int(y/z))

def clean_line(lines, gaps=1):
    lines = np.array(lines)
    xlines = np.ravel(lines[:,0])
    ylines = np.ravel(lines[:,1])

    print("before cleaning :",len(lines))

    tmpX = []
    tmpY = []

    for i in range(len(xlines)):
        if lines[i,5] <= 45:
            for p in range(len(xlines)):
                if xlines[i] > xlines[p]:
                    if xlines[i]-gaps < xlines[p]:
                        if min([i,p]) not in tmpX:
                            tmpX.append(min([i,p]))
                if xlines[i] < xlines[p]:
                    if xlines[i]+gaps > xlines[p]:
                        if min([i,p]) not in tmpX:
                            tmpX.append(min([i,p]))

    for i in range(len(ylines)):
        if lines[i,5] > 45 :
            for p in range(len(ylines)):
                if ylines[i] > ylines[p]:
                    if ylines[i]-gaps < ylines[p]:
                        if min([i,p]) not in tmpY:
                            tmpY.append(min([i,p]))
                if ylines[i] < ylines[p]:
                    if ylines[i]+gaps > ylines[p]:
                        if min([i,p]) not in tmpY:
                            tmpY.append(min([i,p]))

    ret = []
    for i in range(len(lines)):
        if (i not in tmpX) and (i not in tmpY):
            ret.append(lines[i])
    print("after cleaning :",len(ret))

    return np.int32(ret)

def draw_line(image, lines):
    for line in lines:
        cv2.line(image,(line[0],line[1]),(line[2],line[3]),(0,255,0),2)
    return image

def draw_circle(image, circles):
    for circle in circles:
        cv2.circle(image,(circle[0],circle[1]),2,(0,0,255),2)
    return image

def find_anomalies(lines):
    data = normalize(lines[:,4:])
    db = DBSCAN(eps=0.1, min_samples=3).fit(data)
    labels = db.labels_

    anomali_ = []

    for i in range(len(labels)):
        if labels[i] != -1:
            anomali_.append(lines[i])
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    return anomali_
#createBar()

'''
minedge = getBar("Min Edges", "Edges n Lines")
maxedge = getBar("Max Edges", "Edges n Lines")
threshold = getBar("threshold", "Edges n Lines")
minLineLength = getBar("Min Line Length", "Edges n Lines")
maxLineGap = getBar("Max Line Gap", "Edges n Lines")
con = getBar("con", "Edges n Lines")
'''
image = cv2.imread('chessboard/Boards/3.jpg')
image = cv2.resize(image, (int(image.shape[1]*0.2), int(image.shape[0]*0.2)))
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#_, thres = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

edges = cv2.Canny(gray,100,300)

#dst = cv2.cornerHarris(thres,2,3,0.04)
#dst = cv2.dilate(dst,None)
#image[dst>0.01*dst.max()]=[255,0,0]

houghs = cv2.HoughLines(edges,1,np.pi/180,80)

lines = []

if houghs is not None:
    for hough in houghs:
        rho, theta = hough[0]
        a = np.cos(theta)
        b = np.sin(theta)
        
        x0 = a*rho
        y0 = b*rho

        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        
        theta = int(theta*180/np.pi)
        lines.append([x1,y1,x2,y2,rho,theta])

    #image = draw_line(image, clean_line(lines, degree, 2))
    '''
    pair = []
    for i in range(len(lines)):
        for p in range(len(lines)):
            if lines[i] != lines[p]:
                if [i,p] not in pair or [p,i] not in pair:
                    px,py = intersection(lines[i], lines[p])
                    cv2.circle(image, (px,py), 1, (255,0,0), 2)
    '''

#cv2.imshow('thres', thres)
#cv2.imshow('edges', edges)
#cv2.imshow('image', image)

data = find_anomalies(np.array(lines))
data = clean_line(data,3)

image = draw_line(image, data)

intersections = []

for i in range(len(data)):
    for p in range(len(data)):
        l1 = data[i]
        l2 = data[p]
        tmp = find_intersection(l1,l2)
        if tmp[0] != False: 
            intersections.append(tmp)


image = draw_circle(image, intersections)

pts1 = np.float32([[105,275],[445,253],[83,641],[570,582]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(original,M,(300,300))

plt.figure(1)
plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
#plt.scatter(intersections[:][0], intersections[:][1])
plt.figure(2)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

'''
cv2.imshow("image",image)
cv2.waitKey()
if cv2.waitKey(22) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
'''
