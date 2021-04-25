import HandTrackingModule as htm
import cv2 
import time

pTime=0
cTime=0
cap=cv2.VideoCapture(0)
detector=htm.handDetector()

while True:
    success,img=cap.read()         
    img=detector.findHands(img)
    lmsList=detector.findPosition(img)
    if len(lmsList)!=0:
        print(lmsList[8])

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)

    cv2.imshow("Image",img)
    cv2.waitKey(1)