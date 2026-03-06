import cv2
from ultralytics import YOLO


class YoloDetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5, classes=None):
        
        self.model= YOLO(model_path) # path to weights for download
        self.conf_threshold= conf_threshold # confidence threshol
        self.classes= classes # list of classes 


    def detect(self, frame):

        results= self.model(frame, verbose=False)[0]
        detections=[]

        if results.boxes is not None:
            for box in results.boxes:
                conf= float(box.conf[0])
                if conf < self.conf_threshold:
                    continue 
                
                cls= int(box.cls[0])
                if self.classes and cls not in self.classes:
                    continue 
            
                    
                x1,y1,x2,y2= map (int, box.xyxy[0].tolist())
                detections.append([x1,y1,x2,y2,conf, cls])
        
        return detections