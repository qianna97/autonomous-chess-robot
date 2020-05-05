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

def intersection(l1,l2):
    a1,a2 = l1
    b1,b2 = l2

    s = np.vstack([a1,a2,b1,b2])
    h = np.hstack((s, np.ones((4, 1)))) 
    l1 = np.cross(h[0], h[1])           
    l2 = np.cross(h[2], h[3])           
    x, y, z = np.cross(l1, l2)          
    if z == 0:                         
        return (False, False)
    return (int(x/z), int(y/z))

def clean_line(lines, degree, gaps=1):
    lines = np.array(lines)
    xlines = np.ravel(lines[:,0])
    ylines = np.ravel(lines[:,1])
    print("before cleaning :",len(lines))
    tmpX = []
    tmpY = []

    for i in range(len(xlines)):
        if degree[i] <= 45:
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
        if degree[i] > 45 :
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
    return ret

def draw_line(image, lines):
    for line in lines:
        cv2.line(image,(line[0],line[1]),(line[2],line[3]),(0,255,0),2)
    return image

def find_anomalies(data_):
    db = DBSCAN(eps=0.05, min_samples=5).fit(data_)
    labels = db.labels_

    anomali_ = []

    for i in range(len(labels)):
        if labels[i] != -1:
            anomali_.append(data_[i])
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    return anomali_
#createBar()
lines = []
degrees = []

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
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#_, thres = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

edges = cv2.Canny(gray,100,300)

#dst = cv2.cornerHarris(thres,2,3,0.04)
#dst = cv2.dilate(dst,None)
#image[dst>0.01*dst.max()]=[255,0,0]

houghs = cv2.HoughLines(edges,1,np.pi/180,80)

data = []

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
        
        degree = int(theta*180/np.pi)
        data.append([degree,rho])
        #if theta*180/np.pi > 90:
        #cv2.line(image,(x1,y1),(x2,y2),(0,255,0),1)
        degrees.append(degree)
        lines.append([x1,y1,x2,y2])

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



#data = clean_line(lines,degrees,2)
#data = normalize(data, norm='l2', axis=0)
#tes = find_anomalies(np.array(data)[:,0])
#print(tes)
#image = draw_line(image, data)



#kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
#print(len(kmeans.labels_))

plt.figure(1)
plt.scatter(np.arange(0,len(degrees)), np.array(lines)[:,0])
plt.figure(2)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

'''
if cv2.waitKey(22) & 0xFF == ord('q'):
    print(sorted(degree))

    cv2.destroyAllWindows()
'''