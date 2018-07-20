import numpy as np
import cv2
from threading import Thread
from darkflow.net.build import TFNet
import cv2
import pdb
import gc


# this program runs in the background and should connect to another server like module to send events
class Detector():

    def __init__(self,modelPath,weightPath,threshold,borderPadding = 10):
        self.cap = cv2.VideoCapture(0)
        self.tfnet = TFNet({"model": modelPath  , "load": weightPath, "threshold": threshold,"gpu": 1})
        self.isCapturing = True
        self.state = 0 # 0 = detection mode, 1 = tracking mode
        self.target = []
        self.tracker= cv2.TrackerMIL_create()

        self.isTracking = False;
        self.borderPadding = borderPadding
        self.startCapture()


    def getLargestRect(self, rects):
        largestArea = 0
        largestRect =[]
        for rect in rects:
            length = np.abs(rect[0][0] - rect[1][0])
            width = np.abs(rect[0][1] - rect [1][1])
            area = length * width
            if area > largestArea:
                largestArea = area
                largestRect = rect
        self.target = largestRect


    def startCapture(self):
        while(self.isCapturing):
            timer = cv2.getTickCount()

            ret, frame = self.cap.read()
            if(self.state == 1):
                self.track(frame)
            else:
                self.detect(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

        self.cap.release()
        cv2.destroyAllWindows()

    def checkWithinBounds(self,topleft,bottomright):
        return not (topleft[0] < self.borderPadding or topleft[1] < self.borderPadding
        or bottomright[0] > self.cap.get(3) - self.borderPadding
        or bottomright[1] > self.cap.get(4) - self.borderPadding)

    def track(self,frame):
        if self.isTracking:
            self.isTracking, bbox = self.tracker.update(frame)
            topleft = (int(bbox[0]), int(bbox[1]))
            btmright =(int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, topleft, btmright, (255,0,0), 2, 1)
            # if (not self.checkWithinBounds(topleft,btmright)):
            #     self.isTracking = False
        else:
            self.state = 0


    def detect(self,frame):
        results = self.tfnet.return_predict(frame)
        rects = []
        for result in results:
            if result['label']=='person':
                topleft = (result['topleft']['x'],result['topleft']['y'])
                btmright = (result['bottomright']['x'],result['bottomright']['y'])
                rects.append([topleft,btmright])
                cv2.rectangle(frame, topleft, btmright, (0,255,0), 2, 1)
                cv2.putText(frame, result['label'], topleft, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
        if len(rects) != 0:
            self.state = 1
            self.getLargestRect(rects)
            if(self.tracker != None ):
                del self.tracker
                gc.collect()
                self.tracker = cv2.TrackerMIL_create()
            self.isTracking = self.tracker.init(frame,(self.target[0][0],
                                    self.target[0][1],
                                    self.target[1][0],
                                    self.target[1][1]))

    def stopCapture(self):
        self.isCapturing = False

if __name__=="__main__":
    detect = Detector(modelPath = 'cfg/yolov2-tiny.cfg',weightPath = "bin/yolov2-tiny.weights",threshold = 0.1)
