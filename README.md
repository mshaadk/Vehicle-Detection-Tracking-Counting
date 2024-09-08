# YOLOv8 + ByteTrack + Supervision: Object Detection, Tracking, and Counting
This project demonstrates how to combine Ultralytics YOLOv8, ByteTrack, and Supervision to perform object detection, tracking, and counting in a video stream. The setup allows real-time tracking of objects, counting objects that cross a defined line, and saving the results in an output video.

## Project Structure
- **Detection:** We use YOLOv8 for object detection.
- **Tracking:** ByteTrack, integrated via YOLOv8, is used for object tracking.
- **Counting:** Supervision is used to count objects crossing a predefined line in the frame.

## Requirements
To run this project, the following dependencies are required:

```bash
pip install ultralytics
pip install supervision
pip install opencv-python
```

Make sure to have a working Python environment with the required packages installed.

## How to Run
### 1. Detection
We use YOLOv8 to detect objects in a video file and save the results:

```python
from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO('yolov8n.pt')

# Run inference on the video
model.predict(source="testing/d.mp4", save=True, imgsz=320, conf=0.5)
```

### 2. Tracking with ByteTrack
The following code runs object tracking on the same video using ByteTrack:

```python
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Perform tracking using ByteTrack
results = model.track(source="testing/d.mp4", conf=0.3, iou=0.5, save=True, tracker="bytetrack.yaml")
```

### 3. Counting Objects Crossing a Line
We use the Supervision library to count objects that cross a predefined line in the frame. The tracked objects are visualized, and a count of crossed objects is displayed on the video:

```python
import cv2
from collections import defaultdict
import supervision as sv
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Set up video capture and video sink
cap = cv2.VideoCapture("testing/d.mp4")
video_info = sv.VideoInfo.from_video_path("testing/d.mp4")
with sv.VideoSink("output_single_line.mp4", video_info) as sink:
    
    # Define line coordinates
    START = sv.Point(182, 254)
    END = sv.Point(462, 254)

    track_history = defaultdict(lambda: [])
    crossed_objects = {}

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # Run tracking
            results = model.track(frame, classes=[2, 3, 5, 7], persist=True, save=True, tracker="bytetrack.yaml")

            # Extract tracking information
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Visualize detection
            annotated_frame = results[0].plot()

            # Track object movements and detect line crossing
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30: track.pop(0)

                if START.x < x < END.x and abs(y - START.y) < 5 and track_id not in crossed_objects:
                    crossed_objects[track_id] = True

                cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)

            # Draw the line and display the object count
            cv2.line(annotated_frame, (START.x, START.y), (END.x, END.y), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Objects crossed: {len(crossed_objects)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            sink.write_frame(annotated_frame)
        else:
            break

cap.release()
```

### 4. Results
The output video with annotations and object count is saved as output_single_line.mp4. You can check the results by opening the video.

## Project Setup
### Clone the Repository
```bash
git clone https://github.com/mshaadk/Vehicle-Detection-Tracking-Counting.git
cd Vehicle-Detection-Tracking-Counting
```

### Download Pretrained Model
The project uses the YOLOv8n model. You can download the model from the Ultralytics website and place it in the root directory.

## Conclusion
This project provides a seamless way to perform object detection, tracking, and counting using YOLOv8, ByteTrack, and Supervision. It can be adapted for various use cases, including monitoring traffic, tracking people in public spaces, or any other object-tracking tasks.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

## Author
Created by Mohamed Shaad Kunhimohamed

[LinkedIn](https://www.linkedin.com/in/mohamedshaad/)
[GitHub](https://github.com/mshaadk)
