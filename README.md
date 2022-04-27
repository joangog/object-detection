# object-detection
Object detection project using PyTorch on the [COCO 2017](https://cocodataset.org/#download) dataset and the [PWMFD](https://github.com/ethancvaa/Properly-Wearing-Masked-Detect-Dataset) medical mask dataset.

The project implements the two following tasks:

### Evaluation of different object detection models (Faster R-CNN, Mask R-CNN, RetinaNet, SSD, YOLO) for real-time detection on the COCO 2017 dataset.
<p align="center">
  <img width=500 src="https://github.com/joangog/object-detection/blob/main/README_img/coco17_benchmark.png">
</p>

### Real-time detection model of the correct use of medical mask on human faces based on YOLOv5 using the PWMFD dataset.
<p align="center">
  <img width=710 src="https://github.com/joangog/object-detection/blob/main/README_img/mask_demo.gif">
</p>
<p align="center">
  <img width=500 src="https://github.com/joangog/object-detection/blob/main/README_img/mask_examples.png">
</p>

Asset files are found at [object-detection-assets](https://github.com/joangog/object-detection-assets)

Model sources are [Torchvision](https://github.com/pytorch/vision), [YOLOv3](https://github.com/ultralytics/yolov3), [YOLOv5](https://github.com/ultralytics/yolov5), and [YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) repositories.
