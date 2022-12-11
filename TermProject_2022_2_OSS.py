import cv2
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('C:/Users/yckhb/School/OSS/haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('C:/Users/yckhb/School/OSS/haarcascade_eye.xml')

#Functions that run a webcam, add and show emojis to the webcam screen
#Input : image(png), cap(videocapture), flag(int), eyes(list),row(int), col(int) 
#Output : none
#Reference : https://github.com/tanmaya48/OpenCV-puts-glasses-on-face
def photobooth(image,cap,flag,eyes,row,col):
    ret, img = cap.read()
    #Setting Left and right inversion
    img =cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Face recognition using face_cascade
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    ##Look for eyes inside faces
    for (x,y,w,h) in faces:  
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

    if (len(eyes) == 2):   

        flag = 1
        fx = eyes[0,0]+eyes[1,0] 
        fx= int(fx/2 + x)
        fy = eyes[0,1]+eyes[1,1] 
        fy= int(fy/2 + y-100)
        dis = abs(eyes[0,0]-eyes[1,0])

        #Default distance for the scale of eye to eye
        dis_default = 65
        #Rows offset
        dx = 50 
        #Columns offset
        dy = 20 
        ratio = dis/dis_default
        #Shifting offsets to distance
        dx = int(dx*ratio)  
        dy = int(dy*ratio)
        size = (int(col*ratio) , int(row*ratio))

        #Reset the size of the emoji to fit the face size
        img3 = cv2.resize(image,size) 
        rows, cols, channels = img3.shape
        #Make emoji to gray
        img3gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        #Handles thresholds for emoji black/white selection
        ret, mask = cv2.threshold(img3gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

    #flag is used to wait for initial position of emoji to be set
    if flag == 1:
        #Adjust face information
        roi = img[fy+0-dy:fy+rows-dy,fx+0-dx:fx+cols-dx]  
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        img3_fg = cv2.bitwise_and(img3,img3,mask = mask)
        dst = cv2.add(img1_bg,img3_fg)
        #Attach an emoji to the webcam
        img[fy+0-dy:fy+rows-dy, fx+0-dx:fx+cols-dx] = dst
    cv2.imshow('Camera',img) #Show the camera with emoji

#Function in which a user is selected for emoji 
#Input : none
#Output : image(cv2 - png)
def select_emoji():
    #Load the emoji to use
    image1 = cv2.imread('C:/Users/yckhb/School/OSS/emoji/birds.png',cv2.IMREAD_COLOR)
    image2 = cv2.imread('C:/Users/yckhb/School/OSS/emoji/happiness.png',cv2.IMREAD_COLOR)
    image3 = cv2.imread('C:/Users/yckhb/School/OSS/emoji/heart.png',cv2.IMREAD_COLOR)
    image4 = cv2.imread('C:/Users/yckhb/School/OSS/emoji/laugh.png',cv2.IMREAD_COLOR)

    #Save the all emoji to use
    image_list = [image1,image2,image3,image4] 
    #Choose the emoji
    systemnum = int(input("Please select the Emoji  <1 - birds, 2 - happiness, 3 - heart, 4 - laugh> : ")) 
    image = image_list[(systemnum-1)]
    return image

#Main function that execute select_emoji and photobooth
def mainfunc():
    #Load the webcam on laptop(Desktop) 
    cap = cv2.VideoCapture(0)
    flag = 0
    eyes=[]
    image = select_emoji()
    #Set information for the selected emoji
    row, col, channel = image.shape
    #Enter the key 'q' to exit camera
    while cv2.waitKey(33) < 0:
        photobooth(image,cap,flag,eyes,row,col)
    print("Successful connection with webcam and emoji attachment")
    #Release the webcam on laptop(Desktop)
    cap.release()
    #Exit all windows opened through cv2
    cv2.destroyAllWindows()

mainfunc()