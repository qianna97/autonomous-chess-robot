import cv2
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import polynomial_kernel

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

def find_intersection(lines1,lines2):
    ret = []
    for i in range(len(lines1)):
        for p in range(len(lines2)):
            tmp = []
            l1 = lines1[i]
            l2 = lines2[p]

            a1 = [l1[0],l1[1]] 
            a2 = [l1[2],l1[3]]

            b1 = [l2[0],l2[1]] 
            b2 = [l2[2],l2[3]]
            
            s = np.vstack([a1,a2,b1,b2])
            h = np.hstack((s, np.ones((4, 1)))) 
            l1 = np.cross(h[0], h[1])           
            l2 = np.cross(h[2], h[3])           
            x, y, z = np.cross(l1, l2)          
            if z != 0:                         
                tmp = [int(x/z), int(y/z)]
                if (tmp[0] <= width and tmp[0] >= 0) and (tmp[1] <= height and tmp[1] >= 0):
                    ret.append(tmp)

    return ret

def clean_line(lines, h, w, gaps=5):
    print("before cleaning :",len(lines))
    #lines = np.array(lines)
    tmp = []
    for i in range(len(lines)):
        for p in range(len(lines)):
            x,y = find_intersection(lines[i], lines[p])
            if x != False:
                if (x > 0 and y > 0) and (x < width and y < height):
                    #print(x,y)
                    if min([i,p]) not in tmp:
                        tmp.append(min([i,p]))

    '''
    rho = np.ravel(lines[:,4])

    print("before cleaning :",len(lines))

    tmp = []
    pprint(lines)
    for i in range(len(rho)):
        for p in range(len(rho)):
            if rho[i] > rho[p]:
                if rho[i]-gaps < rho[p]:
                    if min([i,p]) not in tmp:
                        tmp.append(min([i,p]))
            if rho[i] < rho[p]:
                if rho[i]+gaps > rho[p]:
                    if min([i,p]) not in tmp:
                        tmp.append(min([i,p]))
    '''
    ret = []
    for i in range(len(lines)):
        if i not in tmp:
            ret.append(lines[i])

    print("after cleaning :",len(ret))
    return np.int16(ret)


def draw_line(image, lines):
    for line in lines:
        cv2.line(image,(line[0],line[1]),(line[2],line[3]),(0,255,0),1)
    return image

def draw_circle(image, circles):
    for i, circle in enumerate(circles):
        cv2.circle(image,(circle[0],circle[1]),1,(0,0,255),2)
        cv2.putText(image, str(i), (circle[0],circle[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA) 
    return image

def find_cluster(lines):
    lines = np.array(lines)
    data = lines[:,4:]

    db = DBSCAN(eps=100, min_samples=5).fit(data)
    labels = db.labels_

    lines1 = []
    lines2 = []

    for i in range(len(labels)):
        if labels[i] == 0:
            lines1.append(lines[i])
        elif labels[i] == 1:
            lines2.append(lines[i])

    under = []
    for i in lines:
        if i[5] <= 45:
            under.append(i)
    
    lines1 = np.array(lines1)
    lines2 = np.array(lines2)

    mean1 = np.mean(np.ravel(lines1[:,5]))
    mean2 = np.mean(np.ravel(lines2[:,5]))

    under = np.array(under)
    
    if mean1 > 135:
        for i in range(len(under)):
            under[i,5] = under[i,5]+mean1 
        lines1 = np.vstack([lines1, under])
    if mean2 > 135:
        for i in range(len(under)):
            under[i,5] = under[i,5]+mean2 
        lines2 = np.vstack([lines2, under])
    
    lines1 = lines1[lines1[:, 4].argsort()]
    lines2 = lines2[lines2[:, 4].argsort()]

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    return np.int16(lines1),np.int16(lines2)

def sort_intersection(intersections, gaps=10):
    rho = intersections

    print("before cleaning :",len(intersections))

    tmp = []

    for i,inter1 in enumerate(intersections):
        for p,inter2 in enumerate(intersections):
            if inter1 != inter2:
                x1 = inter1[0]
                y1 = inter1[1]

                x2 = inter2[0]
                y2 = inter2[1]

                r1 = gaps
                r2 = 1

                distSq = (((x1 - x2)* (x1 - x2))+ ((y1 - y2)* (y1 - y2)))**(.5) 
    
                if (distSq + r2 == r1) or (distSq + r2 < r1): 
                    if min([i,p]) not in tmp:
                            tmp.append(min([i,p]))
            
    ret = []
    for i in range(len(intersections)):
        if i not in tmp:
            ret.append(intersections[i])

    #ret.sort(key=lambda x:np.sqrt(x[0]**2+x[1]**2))
    ret = np.unique(ret, axis=0)

    print("after cleaning :",len(ret))

    return ret

def create_reference(sizeRef):
    sizeSquare = sizeRef/8;
    [xIntersectionsRef, yIntersectionsRef] = np.meshgrid(np.arange(1,10), np.arange(1,10));
    xIntersectionsRef = (xIntersectionsRef-1)*sizeSquare + 1;
    yIntersectionsRef = (yIntersectionsRef-1)*sizeSquare + 1;

    return np.hstack([xIntersectionsRef, yIntersectionsRef])

def find_correspondence(intersections, intersections_ref):
    


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
height = image.shape[0]
width = image.shape[1]

image = cv2.resize(image, (int(width*0.2), int(height*0.2)))
height = image.shape[0]
width = image.shape[1]

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


#cv2.imshow('thres', thres)
#cv2.imshow('edges', edges)
#cv2.imshow('image', image)

lines1, lines2 = find_cluster(lines)

#lines2 = clean_line(lines2, height, width)
image = draw_line(image, lines1)
image = draw_line(image, lines2)

intersections = find_intersection(lines1,lines2)
intersections =sort_intersection(intersections)

image = draw_circle(image, intersections)

intersections_ref = create_reference(300)
'''
pts1 = np.float32([[105,275],[445,253],[83,641],[570,582]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(original,M,(300,300))
'''

#sort_intersection(intersections)

#intersections = np.array(data)

plt.figure(1)
#plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
plt.scatter(np.array(intersections)[:,0], np.array(intersections)[:,1])
plt.figure(2)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

'''
cv2.imshow("image",image)
cv2.waitKey()
if cv2.waitKey(22) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
'''
