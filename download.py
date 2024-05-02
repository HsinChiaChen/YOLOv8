#!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="RtIrGWN0ff4BtnhBijdr")
project = rf.workspace("cheng-kung-uni").project("tree-rrzll")
version = project.version(5)
dataset = version.download("yolov8")

