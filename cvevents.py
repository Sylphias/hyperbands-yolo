import numpy as np
import cv2
from threading import Thread
from darkflow.net.build import TFNet
import cv2
import pdb
import gc
import sys

# class Observer():
#     _observers = []
#     def __init__(self):
#         self._observers.append(self)
#         self._observables = {}
#
#     def observe(self,event_name, callback):
#         self._observables[event_name]= callback

# Begin by specifying a RoI to select structure bounds, uses observer pattern.
# Break bounds into x number of sections, each of these sections will be sent to the LED side
class Detect():
    def __init__(self,numOfSect=8, padding=2):
        self.numOfSect = numOfSect
        video = "test.mp4"
        self.video = cv2.VideoCapture(1)
        self.tfnet = TFNet({"model": 'cfg/yolov2-tiny.cfg'  , "load": "bin/yolov2-tiny.weights", "threshold": 0.1,"gpu":1})
        if not self.video.isOpened():
            print ("Could not open video")
            sys.exit()
        ok, frame = self.video.read()
        if not ok:
            print ('Cannot read video file')
            sys.exit()
        bbox = (287, 23, 86, 320)
        bbox = cv2.selectROI(frame, False)
        self.structBounds = (bbox[0],bbox[1],bbox[2] + bbox[0],bbox[3] + bbox[1])
        # split bbox into sections
        splitWidth = (bbox[2] - (padding*(numOfSect-1)))//numOfSect

        #create bounds for each section
        self.bounds = []
        for x in range(numOfSect):
            topleft = (bbox[0]+x*(splitWidth) + padding*x, bbox[1])
            bottomright = (topleft[0]+splitWidth,bbox[1]+bbox[3])
            self.bounds.append(Bounds(x,topleft,bottomright))
        self.startDetect()


    # for b in self.bounds:
    # cv2.rectangle(frame, b.topLeft, b.btmRight, (255,0,0), 1, 1)

    def startDetect(self):
        while True:
            ok, frame = self.video.read()
            results = self.tfnet.return_predict(frame)
            for b in self.bounds:
                peopleCount = 0
                for result in results:
                    if result['label']=='person':
                        if result['confidence'] < 0.35: continue
                        result_topleft = (result['topleft']['x'],result['topleft']['y'])
                        result_btmright = (result['bottomright']['x'],result['bottomright']['y'])
                        cv2.rectangle(frame, result_topleft, result_btmright, (255,255,0), 2, 1)
                        if b.intersect(result['topleft']['x'],result['topleft']['y'],result['bottomright']['x'],result['bottomright']['y']):
                            peopleCount += 1
                b.update(peopleCount,frame)
                cv2.rectangle(frame, b.topLeft, b.btmRight, (255,0,0), 1, 1)

            cv2.rectangle(frame, (self.structBounds[0:2]), (self.structBounds[2:4]), (0,255,0), 2, 1)

            cv2.imshow("Detecting", frame)
            k = cv2.waitKey(1) & 0xff
            if k == 27 : break


#This class will store a state of the number of people in each section. everytime this number changes, an event is fired off
class Bounds():
    def __init__(self, sectionNum, topLeft, btmRight):
        self.sectionNum = sectionNum
        self.numPeople = 0
        self.topLeft = topLeft
        self.btmRight = btmRight
        self.width = abs(btmRight[0]-topLeft[0])
        self.height = abs(btmRight[1]-topLeft[1])
        self.halfWidth = self.width >> 1
        self.halfHeight = self.height >> 1

    def intersect(self,tlX, tlY, brX, brY):
        boxHalfWidth = abs(tlX - brX) >> 1
        boxHalfHeight = abs(tlY - brY) >> 1
        isInter = (abs(self.topLeft[0]-tlX) <= (self.halfWidth+ boxHalfWidth)) and (abs(self.topLeft[1]-tlY) <= (self.halfHeight + boxHalfHeight))
        return isInter

    def update(self,newNumPeople,frame):
        #count number of people in current frame.
        if self.numPeople < newNumPeople:
            self.fireSection()
            cv2.rectangle(frame, self.topLeft, self.btmRight, (255,255,255), 3, 1)
        self.numPeople = newNumPeople



    def fireSection(self):
        # fires a http request to LED side to trigger an event.
        print(str(self.sectionNum) + "Triggered")
        return






if __name__=="__main__":
    Detect()
