import os
import tensorflow as tf
import cv2

from mrcnn.model import MaskRCNN
from mask_rcnn.scripts.mask_rcnn_visualize import display_instances
from mask_rcnn.scripts.mask_rcnn_config import MaskConfig


sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))  # Detect GPU

# Directory of project
root_dir = os.path.abspath("../")

# Directory of model
model_dir = os.path.join(root_dir, "mask_rcnn")

# Directory of model assets
assets_dir = os.path.join(model_dir, "assets")

# Define model configurations
config = MaskConfig()

# Define model
model = MaskRCNN(mode='inference', model_dir='./log', config=config)

# Load pre-trained COCO weights
model.load_weights(os.path.join(assets_dir, "mask_rcnn_masks_0010.h5"), by_name=True)

# Define COCO class names
class_names = ['BG', 'mask', 'no_mask']

# Load video stream from camera
video = cv2.VideoCapture(0)

# Run detection
while True: # Loop over frames from the video file stream

    # Read next frame of video
    (grabbed, frame) = video.read()
    if not grabbed:
        break
    results = model.detect([frame], verbose=1)

    # Render masked frame
    masked_frame = display_instances(frame, results[0]['rois'], results[0]['masks'], results[0]['class_ids'],
                                class_names, results[0]['scores'])

    cv2.imshow('stream', masked_frame)  # Show rendered frame
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
