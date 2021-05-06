import cv2
import mediapipe as mp
import time
import math

class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionConf=0.5,trackCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionConf=detectionConf
        self.trackConf=trackCon

        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode,self.maxHands,self.detectionConf,self.trackConf)
        
        self.mpDraw=mp.solutions.drawing_utils

        self.tipIds=[4,8,12,16,20]

    def findHands(self,img,draw=True,drawImg=None):
        if drawImg is None:
            drawImg=img
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(drawImg,handLms,self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self,img, handNo=0, draw=False, drawImg=None):
        if drawImg is None:
            drawImg=img
        self.lmList=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                self.lmList.append([id,cx,cy])
                if draw:
                    if id%4==0 or True:
                        cv2.circle(drawImg,(cx,cy),10,(255,0,255),cv2.FILLED)
        return self.lmList

    def boundingBox(self,img,draw=False):
        xList = []
        yList = []
        bbox=[]
        if len(self.lmList)!=0:
            for item in self.lmList:
                xList.append(item[1])
                yList.append(item[2])
            bbox = min(xList), min(yList), max(xList), max(yList)
            if draw:
                cv2.rectangle(img,(bbox[0]-10,bbox[1]-10),(bbox[2]+10,bbox[3]+10),(255,128,25),3)
        return bbox
        

    def fingersUP(self):
        fingers=[]
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for tip in self.tipIds[1:]:
            if self.lmList[tip][2] < self.lmList[tip-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self,p1,p2,img,draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2,y2=self.lmList[p2][1],self.lmList[p2][2]
        cx,cy=(x1+x2)//2,(y1+y2)//2
        if draw:
            cv2.circle(img,(x1,y1),10,(255,0,127),cv2.FILLED)
            cv2.circle(img,(x2,y2),10,(127,0,255),cv2.FILLED)
            cv2.line(img,(x1,y1),(x2,y2),(127,212,34),7)
        length=math.hypot(x2-x1,y2-y1)
        return length, img, [x1,y1,x2,y2,cx,cy]
        
def main():
    pTime=0
    cTime=0
    cap=cv2.VideoCapture(0)
    detector=handDetector()

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


if __name__=="__main__":
    main()
