import cv2
import numpy as np 
from detection.detector import YoloDetector
from tracking.sort import Sort
from boxmot import StrongSORT
import torch


def main(video_path, output_path=None):

    # initializing detector 

    detector= YoloDetector(model_path="yolov8n.pt", conf_threshold=0.5) # only detect person class

    #tracker= Sort(max_age=20,min_hits=3, iou_threshold=0.3) // This is when using our simple SORT tracker


    tracker= StrongSORT(
        model_weights='osnet_x0_25_msmt17.pt', # REID model weights for download
        device= 'cuda' if torch.cuda.is_available() else 'cpu',
        fp16=False, # for faster inference 
        per_class=False, # not using class-specific tracking
        det_thresh=0.5, # detection confidence threshold
        max_dist= 0.2, # maximum cosine distance for REID matching
        min_iou=0.3, # minimum IOU for SORT matching
        mask=False, 
        cmc_method='ecc', # sparse/none
        ecc_mode='affine', # homography is another option 
        ecc_epsilon=1e-4,
        ecc_dt=1.0
    )
    
    
    cap= cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video")
        return 

    # getting video properties
    fps= int(cap.get(cv2.CAP_PROP_FPS))
    width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_path:
        fourcc= cv2.VideoWriter_fourcc(*'mp4v')
        out= cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_id=0
    scale=0.5
    while True:
        ret, frame= cap.read()
        if not ret:
            break 

        #1. Detect objects
        
        sframe=cv2.resize(frame, None, fx=scale, fy=scale) # resizing for faster processing

        detections= detector.detect(sframe)

        original_detections= []
        
        for det in detections:
            x1,y1,x2,y2, conf, cls = det
            x1=int(x1/scale)
            y1=int(y1/scale)
            x2=int(x2/scale)
            y2=int(y2/scale)
            original_detections.append([x1,y1,x2,y2, conf, cls])
 

        if len(original_detections)>0:
            dets=np.array(original_detections) #[:,:5] for simple sort # ignoring class for now

        else:
            dets=np.empty((0,6)) # 5 for simple sort


        # 2. updating the Tracker

        tracked_objects= tracker.update(dets, frame)

        # drawing the results
        

        for det in detections:

            x1,y1,x2,y2, conf, cls = det


            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),2)

            # drawing tracked obects with id

            for track in tracked_objects:
                x1,y1,x2,y2, track_id = map(int,track[:5]) # just track for simple sort
                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            if output_path:
                out.write(frame)
            else:
                cv2.imshow('StrongSORT Tracking', frame)
                if cv2.waitKey(1) & 0xFF ==ord("q"):
                    break
            
            frame_id+=1

    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    if len(sys.argv) <2:
        print("Usage: python main.py <video_path> [output_path]")
        sys.exit(1)
    video_path=sys.argv[1]
    output_path=sys.argv[2] if len(sys.argv) >2 else None

    main(video_path,output_path)