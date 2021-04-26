import cv2
import numpy as np
import HandTrackingModule as htm
import time
import os
import math

wCam=1920
hCam=1080
pTime=0

cap=cv2.VideoCapture(1)
codec = cv2.VideoWriter_fourcc(	'M', 'J', 'P', 'G'	)
cap.set(6, codec)
cap.set(5,30)
cap.set(3,wCam)
cap.set(4,hCam)
print("Camera inicialiced")

folderPath=r"header"
myList=os.listdir(folderPath)
overlayList=[]
for imPath in myList:
    image=cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
header=overlayList[1]
canvasBackground=overlayList[0]

detector=htm.handDetector(maxHands=1, detectionConf=0.75)


canvas=False
mode=0
color=(20,20,20)
size=15
tSelection=[(22,64),(136,178)]
sSelection=[(992,92),(1082,154)]
cSelection=1187
prev_canv_selec=False
tool="c"
xp,yp=(0,0)
imgCanvas=np.zeros((hCam,wCam,3),np.uint8)

while True:
    success,img=cap.read()  
    img=cv2.flip(img,1)
    #img=cv2.resize(img,(wCam,hCam))
    showImg=img.copy()
    if canvas:
        showImg[:,:,:]=canvasBackground[:,:,:]
    showImg[:200,:,:]=header
    cv2.rectangle(showImg,sSelection[0],sSelection[1],(0,0,0),4)
    cv2.rectangle(showImg,tSelection[0],tSelection[1],(0,0,0),4)
    cv2.circle(showImg,(cSelection,120),19,(0,0,0),6)
    cv2.circle(showImg,(cSelection,120),19,(255,255,255),2)
    cv2.circle(showImg,(cSelection,120),19,color,cv2.FILLED)

    img=detector.findHands(img,draw=True, drawImg=showImg)
    lmList=detector.findPosition(img,draw=False)

    if len(lmList)!=0:
        x1,y1=lmList[8][1:]    
        fingers=detector.fingersUP()
        if fingers[1:]==[1,1,0,0]: #two fingers -> Selection mode
            mode=1
            xp,yp=0,0
            cv2.rectangle(showImg,(x1-15,y1-15),(x1+15,y1+15),(0,0,0),cv2.FILLED)
            cv2.rectangle(showImg,(x1-9,y1-9),(x1+9,y1+9),(255,255,255),3)
            if 60<y1<180:
                if 460<x1<530 and prev_canv_selec==False: # canvas selection
                    prev_canv_selec=True
                    canvas= not canvas
                elif 620<x1<812:  # size selection
                    size=50
                    sSelection=[(618,52),(814,181)]
                elif 840<x1<960:
                    size=32
                    sSelection=[(833,80),(960,162)]
                elif 998<x1<1076:
                    size=15
                    sSelection=[(992,92),(1082,154)]
                elif 22<x1<140: #Tool selection
                    tool='c'
                    tSelection=[(22,64),(136,178)]
                elif 185<x1<275: 
                    tool='s'
                    tSelection=[(180,73),(281,170)]
                elif 318<x1<418: 
                    tool='e'
                    tSelection=[(317,77),(417,175)]
                elif 1179<x1<1875:
                    cSelection=x1
                    b,g,r=header[120,x1,:]
                    color=(int(b),int(g),int(r))
            else: prev_canv_selec=False

        elif fingers[1:]==[1,0,0,0]: #One finger -> painting mode
            prev_canv_selec=False
            mode=2
            if tool=='c': #Draw with circle tool
                cv2.circle(showImg,(x1,y1),size,color,cv2.FILLED)
                if y1>200+size:
                    if xp==0 and yp==0:
                        xp,yp=x1,y1
                    cv2.line(imgCanvas,(xp,yp),(x1,y1),color,size*2)
                    xp,yp=x1,y1
            elif tool=='s': #Draw with Square tool
                cv2.rectangle(showImg,(x1-size,y1-size),(x1+size,y1+size),color,cv2.FILLED)
                if y1>200+size:
                    if xp==0 and yp==0:
                        xp,yp=x1,y1
                    cv2.line(imgCanvas,(xp,yp),(x1,y1),color,(2*size))
                    cv2.rectangle(imgCanvas,(x1-size,y1-size),(x1+size,y1+size),color,cv2.FILLED)
                    xp,yp=x1,y1
            elif tool=='e':
                cv2.circle(showImg,(x1,y1),size,(0,0,0),4)
                if y1>200:
                    if xp==0 and yp==0:
                        xp,yp=x1,y1
                    cv2.line(imgCanvas,(xp,yp),(x1,y1),(0,0,0),size*2)
                    xp,yp=x1,y1

        else: 
            mode=0
            xp,yp=0,0
            prev_canv_selec=False

    #showImg=cv2.addWeighted(showImg,0.5,imgCanvas,0.5,0) #Two images transposed
    imgGray=cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _,imgInverse=cv2.threshold(imgGray,10,255,cv2.THRESH_BINARY_INV)
    imgInverse=cv2.cvtColor(imgInverse,cv2.COLOR_GRAY2BGR)
    showImg=cv2.bitwise_and(showImg,imgInverse)
    showImg=cv2.bitwise_or(showImg,imgCanvas)

    cTime=time.time()
    fps=int(1/(cTime-pTime))
    pTime=cTime
    cv2.putText(showImg,"fps: "+str(fps),(1850,1030),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
    cv2.imshow("ARpaint",showImg)
    if (cv2.waitKey(1)  & 0xFF == ord('q')):
        break