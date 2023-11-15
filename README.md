# A Review and Implementation of Object Detection Models and Optimizations for Real-time Medical Mask Detection during the COVID-19 Pandemic
*Note: if you find any problem while running the code, please open a new issue on github. It will be very much appreciated!*

<br>

<p align="center">
  <img width=350 src="https://github.com/joangog/object-detection/blob/main/README_img/inista_award.png">
</p>


This project constitutes the code of our published paper in INISTA 2022, "A Review and Implementation of Object Detection Models and Optimizations for Real-time Medical Mask Detection during the COVID-19 Pandemic", which can be found [here](https://ieeexplore.ieee.org/document/9894232) and [here](https://www.ceid.upatras.gr/webpages/koutsomi/pdf/inista2022.pdf). Please read it for a more detailed analysis of our results.

- The models included in this project are written in PyTorch and evaluated on the [COCO 2017](https://cocodataset.org/#download) dataset and the [PWMFD](https://github.com/ethancvaa/Properly-Wearing-Masked-Detect-Dataset) medical mask dataset.

- The project implements the two following tasks:

<br>

### Evaluation of different object detection models (Faster R-CNN, Mask R-CNN, RetinaNet, SSD, YOLO) for real-time detection on the COCO 2017 dataset.
<p align="center">
  <img width=500 src="https://github.com/joangog/object-detection/blob/main/README_img/coco17_benchmark.png">
</p>

### Real-time detection model of the correct use of medical mask on human faces based on YOLOv5 using the PWMFD dataset.
<p align="center">
  <img width=580 src="https://github.com/joangog/object-detection/blob/main/README_img/mask_demo.gif">
</p>
<p align="center">
  <img width=400 src="https://github.com/joangog/object-detection/blob/main/README_img/mask_examples.png">
</p>

Asset files are found at [object-detection-assets](https://github.com/joangog/object-detection-assets)

Model sources are [Torchvision](https://github.com/pytorch/vision), [YOLOv3](https://github.com/ultralytics/yolov3), [YOLOv5](https://github.com/ultralytics/yolov5), and [YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) repositories.
