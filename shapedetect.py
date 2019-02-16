import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter

img = cv2.imread("./Images/flowchart2.jpg", cv2.IMREAD_GRAYSCALE)
# _, threshold = cv2.threshold(img,128,255,cv2.THRESH_BINARY)
img_xLen, img_yLen = img.shape
#Binarize the image
for cols in range(0,img_yLen-1):
    for rows in range(0, img_xLen-1):
        if(img[rows][cols] < 128):
            img[rows][cols] = 0
        else:
            img[rows][cols] = 255
threshold = img                     #Thresholded image
threshold = 255-threshold           #Invert images
cv2.imshow("Threshold", threshold)

_, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
dst = cv2.GaussianBlur(threshold,(5,5),0)
# GaussianKernel = cv2.getGaussianKernel(ksize=5)
# print(GaussianKernel)


font = cv2.FONT_HERSHEY_COMPLEX

# print(hierarchy.reshape(-1,4))
# print(len(hierarchy.reshape(-1,4)))
hierarchy = hierarchy.reshape(-1,4)
# print(hierarchy)
thres = 500
for cnt in contours:
    # if hierarchy[len(hierarchy)-1][3] == -1:
        approx = cv2.approxPolyDP(cnt, 0.001*cv2.arcLength(cnt, True), True)
        # cv2.drawContours(dst, [approx], 0, (0), 1)
        # if cnt>thres:
        # with np.nditer(cnt, op_flags=['readwrite']) as it:
        # 	for x in it:
        # 		if x[...] > thres:
        # 			x[...] = 0
        # print(type(cnt),cnt, cnt.shape)
        cv2.fillPoly(dst,pts=[approx],color=(255,255,255))
        cv2.polylines(dst,[approx],True,(255,255,255))
        
def findShape(image):
    _, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        cv2.drawContours(image, [approx], 0, (0), 5)
        x = approx.ravel()[0]-10
        y = approx.ravel()[1]-10
        if len(approx) == 3:
            cv2.putText(image, "Triangle", (x,y), font, 1, (0))
        elif len(approx) == 4:
            cv2.putText(image, "quadrilateral", (x,y), font, 1, (0))
        elif len(approx) == 5:
            cv2.putText(image, "Pentagon", (x,y), font, 1, (0))
        # elif 6 < len(approx) < 15:
            # cv2.putText(image, "Ellipse", (x,y), font, 1, (0))
        else:
            cv2.putText(image, "Circle", (x,y), font, 1, (0))
    return image


# kernel = np.ones((5,5), np.float32)/25
kernel = np.ones((30,30),np.float32)
erosion = cv2.erode(dst,kernel,iterations = 1)
# erosion = cv2.morphologyEx(dst,cv2.MORPH_OPEN, kernel)
erosion = 255 - erosion
erode = findShape(erosion)
cv2.imshow("erode",erode)

# cv2.imshow("Threshold", threshold)
# opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
# cv2.imshow("opening", opening)
# dst = cv2.filter2D(threshold, -1, kernel)

cv2.imshow("Original Image", img)
cv2.imshow("Edges detected Image", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()