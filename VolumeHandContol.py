import HandTrackingModule as htm
import cv2 
import time
import numpy as np 
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam=352
hCam=640

cap=cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime=0

detector=htm.handDetector(detectionConf=0.8)

devices=AudioUtilities.GetSpeakers()
interface=devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
volume=cast(interface,POINTER(IAudioEndpointVolume))
minVol=volume.GetVolumeRange()[0]
maxVol=volume.GetVolumeRange()[1]
vol=0
volBar=0
volPer=0

while True:
    success,img=cap.read()  
    img=detector.findHands(img)
    lmList=detector.findPosition(img,draw=False)
    if len(lmList)!=0:
        #print(lmList[4],lmList[8])
        x1,y1=lmList[4][1],lmList[4][2]
        x2,y2=lmList[8][1],lmList[8][2]
        cx,cy=(x1+x2)//2,(y1+y2)//2

        cv2.circle(img,(x1,y1),10,(255,0,127),cv2.FILLED)
        cv2.circle(img,(x2,y2),10,(127,0,255),cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(127,212,34),7)
        

        length=math.hypot(x2-x1,y2-y1)
        cv2.circle(img,(cx,cy),10,(127,34,int(length)),cv2.FILLED)
        vol=np.interp(length,[52,230],[minVol,maxVol])
        print(vol)
        volume.SetMasterVolumeLevel(vol,None)
    
    cv2.rectangle(img,(30,100),(55,350),(198,26,48),3)
    volBar=int(np.interp(vol,[minVol,maxVol],[350,100]))
    cv2.rectangle(img,(30,volBar),(55,350),(198,26,48),cv2.FILLED)
    volPer=int(np.interp(vol,[minVol,maxVol],[0,100]))
    cv2.putText(img,str(volPer)+"%",(30,80),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
    

    cTime=time.time()
    fps=int(1/(cTime-pTime))
    pTime=cTime
    cv2.putText(img,"fps: "+str(fps),(20,40),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)

    cv2.imshow("Image",img)
    if (cv2.waitKey(1)  & 0xFF == ord('q')):
        break