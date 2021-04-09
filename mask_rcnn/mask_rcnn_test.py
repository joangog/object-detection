import os
import tensorflow as tf

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

import cv2
import skimage

from mrcnn.model import MaskRCNN
from mask_rcnn.assets.visualize import display_instances
from mask_rcnn.assets.mask_rcnn_config import MaskConfig

# Directory of project
root_dir = os.path.abspath("../")

# Directory of dataset
dataset_dir = os.path.join(root_dir, "dataset")

# Directory of model
model_dir = os.path.join(root_dir, "mask_rcnn")

# Directory of model assets
assets_dir = os.path.join(model_dir, "assets")

# File name
input_file = 'mask2.jpg'
output_file = 'pedestrians_detected.mp4'

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

# Load test image
# file_names = next(os.walk(image_dir))[2]
# for file_name in file_names:
#     dataset.append(skimage.io.imread(os.path.join(image_dir, file_name)))
image = skimage.io.imread(os.path.join(dataset_dir, input_file))

# Define model configurations
config = MaskConfig()

# Define model
model = MaskRCNN(mode='inference', model_dir='./log', config=config)

# Load pre-trained COCO weights
model.load_weights(os.path.join(assets_dir, "mask_rcnn_masks_0010.h5"), by_name=True)

# Define class names
class_names = ['BG', 'mask', 'no_mask']

# Run detection
results = model.detect([image], verbose=1)

# Render masked image
masked_frame = display_instances(image, results[0]['rois'], results[0]['masks'], results[0]['class_ids'],
                            class_names, results[0]['scores'])

cv2.imshow('result', masked_frame)  # Show rendered image
cv2.waitKey(0)

