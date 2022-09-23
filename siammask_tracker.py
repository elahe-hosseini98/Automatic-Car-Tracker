import numpy as np
from siammask import SiamMask
import cv2
from object_detection import ObjectDetection
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000 

sm = SiamMask()

# Weight files are automatically retrieved from GitHub Releases
sm.load_weights()

cap = cv2.VideoCapture("los_angeles.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('los_angeles_output.mp4', fourcc, fps, (frame_width, frame_height))

# Initialize frame counter
count = 0

# Initialize Object Detection
od = ObjectDetection()

## Detect objects on the first frame by YOLO4
ret, frame = cap.read()
(class_ids, scores, boxes) = od.detect(frame)
all_boxes = np.zeros((len(boxes), 2, 2))

# initialize frame_prev for siammask algorithm
frame_prev = frame

for i, box in enumerate(boxes):
    (x, y, w, h) = box
    all_boxes[i] = np.array([[x, y], [x+w, y+h]])
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cx = int((x + x + w) / 2)
    cy = int((y + y + h) / 2)
    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
    cv2.putText(frame, str(i), (x-2, y-5), 0, 3, (0, 255, 0), 4)
plt.imshow(frame)   
plt.show() 
out.write(frame)

    
# tracking boxes using siammask algorithm
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if count > 0:
        print("Frame #" + str(count))
        
        # siammask tracking for every bounding box in each frame
        frame_copy = frame.copy()
        for i, box_prev in enumerate(all_boxes):
            box, mask = sm.predict(frame_prev, box_prev, frame)
                 
            x1, y1, x2, y2 = int(box[0][0]), int(box[0][1]), int(box[1][0]), int(box[1][1] )
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            (x, y, w, h) = boxes[i]
            
            cv2.rectangle(frame, (cx-int(w/2), cy-int(h/2)), (cx+int(w/2), cy+int(h/2)), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(frame, str(i), (cx-int(w/2)-2, cy-int(h/2)-5), 0, 3, (0, 255, 0), 4)
            
            all_boxes[i] = [[cx-int(w/2), cy-int(h/2)], [cx+int(w/2), cy+int(h/2)]]
        
        frame_prev = frame_copy
        plt.imshow(frame)
        plt.show()
        out.write(frame)
        
    count += 1
   
cap.release()
out.release()    
    
    
    
    
    
    