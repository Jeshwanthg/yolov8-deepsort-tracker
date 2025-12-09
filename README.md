# yolov8-deepsort-tracker
This project implements **real-time object tracking** using:
- **YOLOv8** (Ultralytics) — object detection
- **DeepSORT** — tracking with appearance features (MARS model)
The system detects people across video frames and assigns **consistent track IDs**.
---

🚀 Features
- Real-time multi-object tracking
- Stable tracking using DeepSORT + appearance embeddings
- Unique color-coded bounding boxes
- Works with any YOLOv8 model
- Outputs processed video with tracked IDs
---

📁 Repository Structure
<img width="719" height="611" alt="image" src="https://github.com/user-attachments/assets/b5c4e023-8be9-4474-a35a-5aacda7a84f5" />

---

🚀 Quick Start
# 1. Clone repo
git clone <https://github.com/nwojke/deep_sort.git>
cd yolov8-deepsort

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download pretrained ReID model (if not included) 
mars-small128.pb

# 4. Run tracker script
python main.py ---> Generates an output video with detections

🧠 How It Works
1. YOLOv8 detects objects
- Returns bounding boxes + class + confidence.
  
2. Boxes passed to DeepSORT
DeepSORT uses:
- Kalman Filter (motion prediction)
- Appearance embedding (MARS model)
- Hungarian Algorithm (association)
  
3. Tracks updated
Each person gets:
ID_1, ID_2, ID_3, ...
Even if they leave the frame and come back.
