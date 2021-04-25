import HandTrackingModule as htm
import cv2 
import time
import os
import numpy as np 
import math

wCam=352
hCam=640
pTime=0

cap=cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

folderPath=r"HandTrackingProject\fingers"
myList= os.listdir(folderPath)
overlayList=[]
for imPath in myList:
    image=cv2.imread(f'{folderPath}/{imPath}')
    image=cv2.resize(image,(150,150))
    overlayList.append(image)

detector=htm.handDetector(detectionConf=0.8)
tipIds=[4,8,12,16,20]

while True:
    success,img=cap.read()  
    img=detector.findHands(img)
    lmList=detector.findPosition(img,draw=False)
    if len(lmList)!=0:
        fingers=[]
        if lmList[4][1] < lmList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for tip in tipIds[1:]:
            if lmList[tip][2] < lmList[tip-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        count=fingers.count(1)
        img[10:160,10:160]=overlayList[count]
        cv2.rectangle(img,(180,50),(230,100),(50,50,50),cv2.FILLED)
        cv2.putText(img,str(count),(185,95),cv2.FONT_HERSHEY_PLAIN,4,(200,200,200),2)
        print(count)


    cTime=time.time()
    fps=int(1/(cTime-pTime))
    pTime=cTime
    cv2.putText(img,"fps: "+str(fps),(240,40),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
    cv2.imshow("Image",img)
    if (cv2.waitKey(1)  & 0xFF == ord('q')):
        break