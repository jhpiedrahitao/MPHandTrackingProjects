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
#cap.set(3,wCam)
#cap.set(4,hCam)
pTime=0

detector=htm.handDetector(detectionConf=0.8,maxHands=1)

devices=AudioUtilities.GetSpeakers()
interface=devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
volume=cast(interface,POINTER(IAudioEndpointVolume))
minVol=volume.GetVolumeRange()[0]
maxVol=volume.GetVolumeRange()[1]
vol=0
volPer=int(volume.GetMasterVolumeLevelScalar()*100)
fixedVol=volPer
volBar = int(np.interp(volPer, [0, 100], [350, 100]))
area=0
smoothness = 10

while True:
    success,img=cap.read()  
    img=detector.findHands(img)
    lmList=detector.findPosition(img,draw=False)
    cv2.rectangle(img, (30, 100), (55, 350), (198, 26, 48), 3)
    cv2.rectangle(img, (30, volBar), (55, 350), (198, 26, 48), cv2.FILLED)
    cv2.putText(img, str(fixedVol)+"%", (30, 80),cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
    if len(lmList)!=0:
        bbox=detector.boundingBox(img)
        area=(bbox[2]-bbox[0])*(bbox[3]-bbox[1])//100
        if 300<area<1000:
            length, ing ,lineInfo = detector.findDistance(4,8,img)
            cx, cy = lineInfo[4], lineInfo[5]
            cv2.circle(img, (cx, cy), 10, (127, 34, int(length)), cv2.FILLED)

            volBar = int(np.interp(length, [30, 230], [350, 100]))
            volPer = int(np.interp(length, [30, 230], [0, 100]))
            volPer=smoothness*round(volPer/smoothness)

            fingers=detector.fingersUP()
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(volPer/100, None)
                fixedVol=volPer
                print(fixedVol)
                cv2.circle(img, (cx, cy), 20, (50, 200, 50), cv2.FILLED)
                cv2.putText(img, str(fixedVol)+"%", (30, 80),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (50, 255, 50), 2)
                #time.sleep(0.1)
    
    cTime=time.time()
    fps=int(1/(cTime-pTime))
    pTime=cTime
    cv2.putText(img,"fps: "+str(fps),(20,40),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)

    cv2.imshow("Image",img)
    if (cv2.waitKey(1)  & 0xFF == ord('q')):
        break
