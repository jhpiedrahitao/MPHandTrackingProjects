#hand tracking and gesture mouse
import cv2 
import numpy as np
import HandTrackingModule as htm
import time
import autopy
from autopy.mouse import Button
from playsound import playsound
import os

wCam=1280
hCam=720
wScr,hScr=autopy.screen.size()
frameReduction=100
smoothening=8

activatedAudioPath = os.path.join("audios","mouseActive.mp3")
deactivatedAudioPath = os.path.join("audios","mouseNotActive.mp3")

cap=cv2.VideoCapture(1)
cap.set(3,wCam)
cap.set(4,hCam)

pTime=0
mode="n"
vMouseActive=False
vMouseActivated=False
lastMode="n"
pLocX,pLocY=0,0
cLocX,cLocY=0,0

detector=htm.handDetector(detectionConf=0.8,maxHands=1)

while True:
    success,img=cap.read()  
    img=cv2.flip(img,1)
    img=detector.findHands(img)
    lmList=detector.findPosition(img,draw=False)
    if len(lmList)!=0:
        bbox=detector.boundingBox(img,draw=True)
        if (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) > (wCam*hCam/3): 
            if vMouseActivated==False:
                vMouseActivated=True
                vMouseActive = not vMouseActive
                if vMouseActive: playsound(activatedAudioPath)
                else: playsound(deactivatedAudioPath)
        else: vMouseActivated=False
        

        x1,y1=lmList[5][1:]
        xS=np.interp(x1,(frameReduction,wCam-frameReduction),(0,wScr-1),0,wScr-1)
        yS=np.interp(y1,(frameReduction,hCam-frameReduction),(0,hScr-1),0,hScr-1)
        cLocX=pLocX+(xS-pLocX)/smoothening
        cLocY=pLocY+(yS-pLocY)/smoothening
        
        fingersUP=detector.fingersUP()
        if vMouseActive:
            if fingersUP[1:]==[1,1,1,1]: mode="m"
            elif fingersUP[:]==[1,0,0,0,0]: mode="l"
            elif fingersUP[1:]==[0,1,1,1]: mode="t"
            elif fingersUP[1:]==[1,1,0,0]: mode="r"
            elif fingersUP[:]==[0,0,0,0,0]: mode="s"
            else: mode="n"
        else: mode="n"
    else: mode="n"

    if mode!="n":
        if mode!="l":
            #autopy.mouse.smooth_move(cLocX,cLocY)
            autopy.mouse.move(cLocX,cLocY)
        pLocX,pLocY = cLocX,cLocY
        if mode=='t': 
            if lastMode!="t":
                autopy.mouse.toggle(button=Button.LEFT,down=True)
        elif mode=="l":
            if lastMode!="l":
                autopy.mouse.click(button=Button.LEFT)
        elif mode=="r":
            if lastMode!="r":
                autopy.mouse.click(button=Button.RIGHT)
        elif mode=='s': 
            if lastMode!="s":
                autopy.mouse.toggle(button=Button.MIDDLE,down=True)
        else:
            autopy.mouse.toggle(button=Button.LEFT,down=False)
            autopy.mouse.toggle(button=Button.MIDDLE,down=False)
    else: 
        autopy.mouse.toggle(button=Button.LEFT,down=False)
        autopy.mouse.toggle(button=Button.MIDDLE,down=False)
    lastMode=mode

    cTime=time.time()
    fps=int(1/(cTime-pTime))
    pTime=cTime
    cv2.putText(img,"fps: "+str(fps),(20,40),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)

    cv2.imshow("Image",img)
    if (cv2.waitKey(1)  & 0xFF == ord('q')):
        break