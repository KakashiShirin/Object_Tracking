import cv2
from tracker import *

# creating tracker object.
tracker= EuclideanDistance()

cap= cv2.VideoCapture("C:\Testing\Object_Tracking\sample.mp4")


#object detection for stable camers

object_detector= cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=20)
# using loop for reading video frame by frame

while True:
    ret, frame= cap.read()

    h,w,_=frame.shape
    
    #print (h,w,_)
    # defining region of interest.

    #object detections
    roi= frame[600:,250:1500]   

    mask= object_detector.apply(roi)
    _, mask = cv2.threshold (mask, 253,255,cv2.THRESH_BINARY)
    contours,_= cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections=[]
    
    
    for cnt in contours:

        area= cv2.contourArea(cnt)
        if area >1600:
            #cv2.drawContours(roi, [cnt], -1, (0,255,0), 2)
            x,y,w,h = cv2. boundingRect(cnt)
            detections.append([x,y,w,h])
            

    # object tracking

    box_ids=tracker.update(detections)
    
    for bix_id in box_ids:
        x,y,w,h,id= bix_id
        cv2.rectangle(roi, (x,y),(x+w, y+h), (0,255,0),2)
        cv2.putText(roi, str(id),(x,y-15), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),2)
            
    print(box_ids)

    print(detections)
    cv2.imshow("roi", roi)
    cv2.imshow("Mask", mask)
    #cv2.imshow("Frame",frame)

    key= cv2.waitKey(30)
    if key == 27: # when user clicks escape key it ends the code.
        break

cap.release()
cv2.destroyAllWindows()