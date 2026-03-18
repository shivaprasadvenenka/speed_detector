import cv2
import time
import os
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
video_path = "mv2.mp4"
cap = cv2.VideoCapture(video_path)

plate_numbe= os.path.splitext(video_path)[0]

line1_y = 250
line2_y = 350

distance_meters = 10

vehicle_times = {}
vehicle_speeds = {}
previous_positions = {}

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame,(1000,600))

    results = model.track(frame, persist=True, verbose=False)[0]

    if results.boxes.id is not None:

        boxes = results.boxes.xyxy
        ids = results.boxes.id.int().tolist()
        classes = results.boxes.cls

        for box,track_id,cls in zip(boxes,ids,classes):

            x1,y1,x2,y2 = map(int,box)
            label = model.names[int(cls)]

            if label in ["car","truck","bus","motorbike"]:

                cx = int((x1+x2)/2)
                cy = int((y1+y2)/2)

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                if track_id in previous_positions:

                    prev_y = previous_positions[track_id]

                    # crossed line1
                    if prev_y < line1_y and cy >= line1_y:
                        vehicle_times[track_id] = time.time()

                    # crossed line2
                    if prev_y < line2_y and cy >= line2_y:

                        if track_id in vehicle_times and track_id not in vehicle_speeds:

                            time_diff = time.time() - vehicle_times[track_id]

                            speed = (distance_meters/time_diff)*3.6

                            vehicle_speeds[track_id] = int(speed)

                previous_positions[track_id] = cy

                if track_id in vehicle_speeds:

                    text = f"{plate_number} : {vehicle_speeds[track_id]} km/h"

                    cv2.putText(frame,text,(x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),2)

    cv2.line(frame,(0,line1_y),(1000,line1_y),(255,0,0),2)
    cv2.line(frame,(0,line2_y),(1000,line2_y),(255,0,0),2)

    cv2.imshow("Vehicle Speed Detection",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()