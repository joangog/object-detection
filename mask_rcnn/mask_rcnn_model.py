import os, sys
import tensorflow as tf

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide TensorFlow warnings
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

import skimage.io
import cv2
import matplotlib as plt

from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn import utils
from mrcnn import visualize
from mask_rcnn.visualize import display_instances

# Directory of the project
root_dir = os.path.abspath("../")
sys.path.append(root_dir)  # To find local version of the library

# Directory of test images
image_dir = os.path.join(root_dir, "images")

# Directory of model
model_dir = os.path.join(root_dir, "mask_rcnn")

# File name
input_file = 'pedestrians.mp4'
output_file = 'pedestrians_detected.mp4'

# Load video stream from camera
video = cv2.VideoCapture(0)

# # Load video
# video = cv2.VideoCapture(os.path.join(image_dir, input_file))
# frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
# frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # approximate frame count
#
# print("Total frames to render: " + str(frame_count))

# # Initialize video writer
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# writer = cv2.VideoWriter(os.path.join(image_dir, output_file), fourcc, 30, (frame_width, frame_height), True)

# # Load test images
# images = []
# file_names = next(os.walk(image_dir))[2]
# for file_name in file_names:
#     images.append(skimage.io.imread(os.path.join(image_dir, file_name)))

# Define model configurations
class MyConfig(Config):
    NAME = "coco"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80  # COCO has 80 classes

# Define model
model = MaskRCNN(mode='inference', model_dir='./log', config=MyConfig())

# Load pre-trained COCO weights
model.load_weights(os.path.join(model_dir, "mask_rcnn_coco.h5"), by_name=True)

# Define COCO class names
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Run detection
while True: # loop over frames from the video file stream

    # Read next frame of video
    (grabbed, frame) = video.read()
    if not grabbed:
        break
    results = model.detect([frame], verbose=1)

    # Render masked frame
    masked_frame = display_instances(frame, results[0]['rois'], results[0]['masks'], results[0]['class_ids'],
                                class_names, results[0]['scores'])

    cv2.imshow('stream', masked_frame)  # show rendered frame
    cv2.waitKey(1)
    # writer.write(masked_frame)  # write each frame in file

video.release()
cv2.destroyAllWindows()
