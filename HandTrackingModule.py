import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionConf=0.5,trackCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionConf=detectionConf
        self.trackConf=trackCon

        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode,self.maxHands,self.detectionConf,self.trackConf)
        
        self.mpDraw=mp.solutions.drawing_utils

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

    def findPosition(self,img, handNo=0, draw=True, drawImg=None):
        if drawImg is None:
            drawImg=img
        lmList=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw:
                    if id%4==0 or True:
                        cv2.circle(drawImg,(cx,cy),10,(255,0,255),cv2.FILLED)
        return lmList
        
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