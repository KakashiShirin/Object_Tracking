import numpy as np 
from scipy.optimize import linear_sum_assignment 
from filterpy.kalman import KalmanFilter  


def iou(bb_test, bb_gt):

    # finding intersection-over-union between 2 boxesin the form [x1,y1,x2,y2]

    xx1=max(bb_test[0], bb_gt[0])
    yy1= max(bb_test[1], bb_gt[1])
    xx2= min(bb_test[2], bb_gt[2])
    yy2= min(bb_test[3], bb_gt[3])

    w= max(0, xx2-xx1)
    h= max(0, yy2-yy1)

    wh= w*h 

    area_test= ((bb_test[2]- bb_test[0])* (bb_test[3]-bb_test[1]))
    area_gt= ((bb_gt[2]-bb_gt[0])*(bb_gt[3]- bb_gt[1]))

    return wh/ (area_test+ area_gt- wh)


def convert_bbox_to_z(bbox):
    # converting bbox to kalman filter state, ie [x,y,scale, aspect_ratio]^T 

    w= bbox[2]-bbox[0]
    h= bbox[3]-bbox[1]

    x= bbox[0] + w/2
    y= bbox[1] + h/2
    s= w*h 
    r= w/ float(h)

    return np.array([x,y,s,r]).reshape((4,1))


def convert_x_to_bbox(x, score=None):

    # converting kalman state to bbox 

    w= np.sqrt(x[2]*x[3])
    h= x[2]/w
    x1= x[0]-w/2
    y1= x[1]-h/2
    x2= x[0]+w/2
    y2= x[1]+h/2

    if score is None:
        return np.array([x1,y1,x2,y2]).reshape((1,4))
    else:
        return np.array([x1,y1,x2,y2, score]).reshape((1,5))
    



class KalmanTracker:

    count=0

    def __init__ (self, bbox):

        # initializing tracker with bounding box
        self.kf= KalmanFilter(dim_x=7, dim_z=4)
        
        self.kf.F= np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        
        self.kf.H= np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
        
        self.kf.R[2:,2:]*=10.
        self.kf.P[4:,4:]*=1000.   # high uncertainity for unobservable initial velocities
        self.kf.P*=10.
        self.kf.Q[-1,-1]*=0.01
        self.kf.Q[4:,4:]*=0.01


        self.kf.x[:4]= convert_bbox_to_z(bbox)
        self.time_since_update=0
        self.id = KalmanTracker.count
        KalmanTracker.count+=1
        self.history=[]
        self.hits=0
        self.hit_streak=0
        self.age=0 


    def update(self,bbox):

        # updating state vector with observation bbox

        self.time_since_update=0
        self.history=[]
        self.hits+=1
        self.hit_streak+=1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):

        if (self.kf.x[6] + self.kf.x[2])<=0:
            self.kf.x[6]*=0.0
        self.kf.predict()
        self.age+=1
        
        if self.time_since_update>0:
            self.hit_streak=0
        self.time_since_update+=1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)
    

def detect_association_to_tracker(detections, trackers, iou_threshold=0.3):
    # assigning detections to tracked objects 

    if len(trackers)==0:
        return np.empty((0,2), dtype=int), np.arange(len(detections)), np.empty((0,5), dtype=int)
    
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t]= iou(det, trk)

    matched_indices= linear_sum_assignment(-iou_matrix) # minimizing cost
    matched_indices= np.array(list(zip(*matched_indices)))

    unmatched_detections= []

    for d,det in enumerate(detections):
        if d not in matched_indices[:,0]:
            unmatched_detections.append(d)
    
    unmatched_trackers= []
    
    for t,trk in enumerate(trackers):
        if t not in matched_indices[:,1]:
            unmatched_trackers.append(t)

    # filtering matches

    matches=[]

    for m in matched_indices:
        if iou_matrix[m[0], m[1]]< iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        
        else:
            matches.append(m.reshape(1,2))

    if len(matches)==0:
        matches= np.empty((0,2), dtype=int)
    else:
        matches=np.concatenate(matches, axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)




#SORT

class Sort:
    def __init__ (self, max_age=1, min_hits=3, iou_threshold=0.3):

        self.max_age=max_age
        self.min_hits= min_hits
        self.iou_threshold= iou_threshold
        self.trackers=[]
        self.frame_count=0

    def update(self,dets):

        self.frame_count+=1

        trks= np.zeros((len(self.trackers),5))
        to_del=[]
        ret=[]

        for t,trk in enumerate(trks):
            pos= self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]

            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks= np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets,unmatched_trks= detect_association_to_tracker(dets, trks, self.iou_threshold)

        # updating matched trackers with their detections

        for m in matched:
            self.trackers[m[1]].update(dets[m[0],:4])

        # creating new trackers for unmatched

        for i in unmatched_dets:
            trk= KalmanTracker(dets[i,:4])
            self.trackers.append(trk)

        i= len(self.trackers)
        for trk in reversed(self.trackers):
            d= trk.get_state()[0]

            if (trk.time_since_update <1) and (trk.hit_streak >= self.min_hits or self.frame_count<=self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1,-1)) 

                i-=1

                # removing dead trackers 
                if trk.time_since_update > self.max_age:
                    self.trackers.pop(i)
        
        if len(ret)>0:
            return np.concatenate(ret)
        return np.empty((0,5))