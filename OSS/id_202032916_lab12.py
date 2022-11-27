import math
import cv2

src = cv2.imread("C:/Users/yckhb/Downloads/862227512_8de1721dea_o.jpg")
dst = src.copy()
gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray,5000,1500,apertureSize=5,L2gradient=True)
lines = cv2.HoughLinesP(canny,0.8,math.pi/180,90,minLineLength=10,maxLineGap=50)
circle = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,800,param1=200,param2=20,minRadius=90,maxRadius=110)

for i in lines:
    cv2.line(dst,(int(i[0][0]),int(i[0][1])),(int(i[0][2]),int(i[0][3])),(0,0,255),2)

for c in circle[0]:
    x,y,r = int(c[0]), int(c[1]),int(c[2])
    cv2.circle(dst,(x,y),r,(255,255,255),5)

cv2.imshow("Deteced",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()