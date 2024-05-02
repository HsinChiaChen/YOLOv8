from ultralytics import YOLO
import os
import cv2

dirPath = os.path.dirname(os.path.realpath(__file__))
print(dirPath)

# Load a model
# Load a model
model = YOLO('yolov8s.pt')  # load an official model

#########################################################
# Object Detect
## forest_detect_0312_1 : Fram
## Fram_env : Fram
## detect_1125 : NCKU cone
## gazebo_detext_170124 : gazebo

# Instance Segmentation
## school_seg_0312_2 : HCJH school
## school_seg_0429: NCKU school
## forest_seg_0312_2 : Fram
###########################################################

model = YOLO(dirPath+'/weight/forest_detect_0312_1.pt')  # load a custom trained
model.predict(dirPath + '/original_data/video/Field_01.mp4', save=True)
