from ultralytics import YOLO
import os
import cv2

dirPath = os.path.dirname(os.path.realpath(__file__))
print(dirPath)

# Load a model
# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official model
model = YOLO(dirPath + '/weight/NCKU_seg_0502.pt')  # load a custom trained
model.predict(dirPath + '/original_data/video/NCKU/NCKU_tree_0430_2.mp4', save=True)
