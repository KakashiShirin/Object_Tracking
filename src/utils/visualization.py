import cv2

def draw_detections(frame, detections, color=(255,0,0), show_conf=True):

    for det in detections:

        if len(detections)>=5:
            x1,y1,x2,y2,conf =det[:5]

            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
            if show_conf:
                cv2.putText(frame, f"{conf:.2f}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame


def draw_tracks(frame, tracks, color=(0,255,0), show_id=True):

    for track in tracks:
        x1,y1,x2,y2,track_id = map(int, track[:5])
        cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
        if show_id:
            cv2.putText(frame, label, (x1, y2+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame