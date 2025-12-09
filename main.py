"""
main.py
-------
Runs YOLOv8 + DeepSORT for real-time multi-object tracking.

Pipeline:
1. Load video
2. Run YOLOv8 for object detection
3. Pass detections to DeepSORT for tracking
4. Draw tracked bounding boxes with unique IDs
5. Save output video

Author: Jeshwanth Ganesh
"""
import os
import random
import cv2
from ultralytics import YOLO
from deep_sort.deep_sort.deep_sort_tracker import Tracker


# ----------------------------
# Configuration
# ----------------------------
video_path = "people.mp4"
video_out_path = "out.mp4"

# Load input video
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

# Prepare output video
cap_out = cv2.VideoWriter(
    video_out_path,
    cv2.VideoWriter_fourcc(*'MP4V'),
    cap.get(cv2.CAP_PROP_FPS),
    (frame.shape[1], frame.shape[0])
)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize DeepSORT tracker
tracker = Tracker()

# Random colors for track IDs
colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) 
          for _ in range(50)]

detection_threshold = 0.5


# ----------------------------
# Processing Loop
# ----------------------------
while ret:

    # 1. Run YOLOv8 on the frame
    results = model(frame)

    for result in results:
        detections = []

        # Extract bounding boxes
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])

        # 2. Update DeepSORT with YOLO detections
        tracker.update(frame, detections)

        # 3. Draw tracked boxes
        for track in tracker.tracks:
            x1, y1, x2, y2 = track.bbox
            track_id = track.track_id

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                colors[track_id % len(colors)],
                3
            )
            cv2.putText(
                frame,
                f"ID: {track_id}",
                (int(x1), int(y1)-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                colors[track_id % len(colors)],
                2
            )

    cap_out.write(frame)
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()
